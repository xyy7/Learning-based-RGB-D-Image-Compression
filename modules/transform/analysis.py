import numpy as np
import torch
import torch.nn as nn
from compressai.layers import AttentionBlock
from modules.layers.conv import conv, conv1x1, conv3x3, deconv
from modules.layers.res_blk import *

from .attention import *


class AnalysisTransform(nn.Module):
    def __init__(self, N, M, channel=3):
        super().__init__()
        self.analysis_transform = nn.Sequential(
            ResidualBlockWithStride(channel, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, M, stride=2),
        )

    def forward(self, x):
        x = self.analysis_transform(x)
        return x


class AnalysisTransformEX(nn.Module):
    def __init__(self, N, M, ch=3, act=nn.ReLU):
        super().__init__()
        self.analysis_transform = nn.Sequential(
            conv(ch, N),  # 通过卷积进行下采样
            ResidualBottleneck(N, act=act),  # 通过残差块来增强特征
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            conv(N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            AttentionBlock(N),
            conv(N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            conv(N, M),
            AttentionBlock(M),  # 通过通道注意力，进一步增强特征
        )

    def forward(self, x):
        x = self.analysis_transform(x)
        return x


# 接受rgb和depth作为输入输出，但需要进行交互
class AnalysisTransformEXSingle(nn.Module):
    def __init__(self, N, M, act=nn.ReLU):
        super().__init__()
        self.rgb_analysis_transform = nn.Sequential(
            conv(3, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            bi_spf_single(N),
            conv(N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            AttentionBlock(N),
            bi_spf_single(N),
            conv(N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            bi_spf_single(N),
            conv(N, M),
            AttentionBlock(M),
        )

        self.depth_analysis_transform = nn.Sequential(
            conv(1, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            nn.Identity(),
            conv(2 * N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            AttentionBlock(N),
            nn.Identity(),
            conv(2 * N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            nn.Identity(),
            conv(2 * N, M),
            AttentionBlock(M),
        )

    def forward(self, rgb, depth):
        rgb_y = rgb
        depth_y = depth
        for num, (rgb_bk, depth_bk) in enumerate(zip(self.rgb_analysis_transform, self.depth_analysis_transform)):
            if isinstance(rgb_bk, bi_spf_single):
                depth_f = rgb_bk(rgb_y, depth_y)
                depth_y = torch.cat((depth_y, depth_f), dim=-3)
            else:
                rgb_y = rgb_bk(rgb_y)
                depth_y = depth_bk(depth_y)

        return rgb_y, depth_y


# 接受rgb和depth作为输入输出，但需要进行交互
class AnalysisTransformEXcross(nn.Module):
    def __init__(self, N, M, act=nn.ReLU):
        super().__init__()
        self.rgb_analysis_transform = nn.Sequential(
            conv(3, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            bi_spf(N),
            conv(2 * N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            AttentionBlock(N),
            bi_spf(N),
            conv(2 * N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            bi_spf(N),
            conv(2 * N, M),
            AttentionBlock(M),
        )

        self.depth_analysis_transform = nn.Sequential(
            conv(1, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            nn.Identity(),
            conv(2 * N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            AttentionBlock(N),
            nn.Identity(),
            conv(2 * N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            nn.Identity(),
            conv(2 * N, M),
            AttentionBlock(M),
        )

    def forward(self, rgb, depth):
        rgb_y = rgb
        depth_y = depth
        for num, (rgb_bk, depth_bk) in enumerate(zip(self.rgb_analysis_transform, self.depth_analysis_transform)):
            if isinstance(rgb_bk, bi_spf):
                depth_y = depth_bk(depth_y)
                rgb_f, depth_f = rgb_bk(rgb_y, depth_y)
                rgb_y = torch.cat((rgb_y, rgb_f), dim=-3)
                depth_y = torch.cat((depth_y, depth_f), dim=-3)
            else:
                rgb_y = rgb_bk(rgb_y)
                depth_y = depth_bk(depth_y)

        return rgb_y, depth_y


#################### HyperAnalysis ###############################


class HyperAnalysis(nn.Module):
    """
    Local reference
    """

    def __init__(self, M=192, N=192):
        super().__init__()
        self.M = M
        self.N = N
        self.reduction = nn.Sequential(
            conv3x3(M, N),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            conv3x3(N, N, stride=2),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            conv3x3(N, N, stride=2),
        )

    def forward(self, x):
        x = self.reduction(x)

        return x


class HyperAnalysisEX(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.reduction = nn.Sequential(conv3x3(M, N), act(), conv(N, N), act(), conv(N, N))

    def forward(self, x):
        x = self.reduction(x)
        return x

class HyperAnalysisEXcro(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.rgb_reduction = nn.Sequential(conv3x3(M, N), act(), conv(N, N), act(), conv(N, N))
        self.depth_reduction = nn.Sequential(conv3x3(M, N), act(), conv(N, N), act(), conv(N, N))

    def forward(self, rgb, depth):
        rgb = self.rgb_reduction(rgb)
        depth = self.depth_reduction(depth)
        return rgb, depth

class HyperAnalysisEXcross(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.rgb_reduction = nn.Sequential(conv3x3(M, N), act(), conv(N, N), act(), conv(N, N))
        self.depth_reduction = nn.Sequential(conv3x3(M, N), act(), conv(N, N), act(), conv(N, N))

    def forward(self, rgb, depth):
        rgb = self.rgb_reduction(rgb)
        depth = self.depth_reduction(depth)
        return rgb, depth
