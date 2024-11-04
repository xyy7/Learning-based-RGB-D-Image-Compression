import torch
import torch.nn as nn
from compressai.layers import AttentionBlock, subpel_conv3x3
from modules.layers.conv import conv, conv1x1, conv3x3, deconv
from modules.layers.res_blk import (ResidualBlock, ResidualBlockUpsample,
                                    ResidualBottleneck)

from .attention import *
from .spatialAligner import Spatial_aligner


class SynthesisTransform(nn.Module):
    def __init__(self, N, M, channel=3):
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            ResidualBlock(M, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, channel, 2),
        )

    def forward(self, x):
        x = self.synthesis_transform(x)

        return x


class SynthesisTransformEX(nn.Module):
    def __init__(self, N, M, ch=3, act=nn.ReLU, return_mid=False) -> None:
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            AttentionBlock(N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, ch),
        )
        self.return_mid = return_mid

    def forward(self, x):
        num = 0
        if self.return_mid:
            for bk in self.synthesis_transform:
                x = bk(x)
                if isinstance(bk, nn.ConvTranspose2d):
                    if num == 0:
                        up1 = x
                    if num == 1:
                        up2 = x
                    if num == 2:
                        up3 = x
                    num += 1
            return x, up1, up2, up3

        x = self.synthesis_transform(x)
        return x


# Luguo's Spatial_aligner
class SynthesisTransformPlus(nn.Module):
    def __init__(self, N, M, ch=3, act=nn.ReLU) -> None:
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            AttentionBlock(N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, ch),
        )
        self.sp1 = Spatial_aligner()
        self.sp2 = Spatial_aligner()
        self.sp3 = Spatial_aligner()

    def forward(self, x, up1, up2, up3):
        num = 0
        for bk in self.synthesis_transform:
            x = bk(x)
            if isinstance(bk, nn.ConvTranspose2d):
                if num == 0:
                    x = self.sp1(x, up1)
                if num == 1:
                    x = self.sp2(x, up2)
                if num == 2:
                    x = self.sp3(x, up3)
                num += 1
        return x


# 接受rgb和depth作为输入输出，但并没有进行交互
class SynthesisTransformEXcro(nn.Module):
    def __init__(self, N, M, act=nn.ReLU):
        super().__init__()
        self.rgb_synthesis_transform = SynthesisTransformEX(N, M, ch=3)
        self.depth_synthesis_transform = SynthesisTransformEX(N, M, ch=1)

    def forward(self, rgb, depth):
        rgb = self.rgb_synthesis_transform(rgb)
        depth = self.depth_synthesis_transform(depth)
        return rgb, depth


class SynthesisTransformEXcross(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.rgb_synthesis_transform = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N),
            bi_spf(N),
            ResidualBottleneck(2 * N, N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            AttentionBlock(N),
            bi_spf(N),
            ResidualBottleneck(2 * N, N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            bi_spf(N),
            ResidualBottleneck(2 * N, N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, 3),
        )
        # 使用identity进行占位
        self.depth_synthesis_transform = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N),
            nn.Identity(),
            ResidualBottleneck(2 * N, N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            AttentionBlock(N),
            nn.Identity(),
            ResidualBottleneck(2 * N, N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            nn.Identity(),
            ResidualBottleneck(2 * N, N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, 1),
        )

    def forward(self, rgb, depth):
        rgb_hat = rgb
        depth_hat = depth
        for num, (rgb_bk, depth_bk) in enumerate(zip(self.rgb_synthesis_transform, self.depth_synthesis_transform)):
            if isinstance(rgb_bk, bi_spf):
                depth_hat = depth_bk(depth_hat)
                rgb_f, depth_f = rgb_bk(rgb_hat, depth_hat)
                rgb_hat = torch.cat((rgb_hat, rgb_f), dim=-3)
                depth_hat = torch.cat((depth_hat, depth_f), dim=-3)
            else:
                rgb_hat = rgb_bk(rgb_hat)
                depth_hat = depth_bk(depth_hat)

        return rgb_hat, depth_hat

class SynthesisTransformEXSingle(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.rgb_synthesis_transform = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N),
            bi_spf_single(N),
            ResidualBottleneck(N, N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            AttentionBlock(N),
            bi_spf_single(N),
            ResidualBottleneck(N, N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            bi_spf_single(N),
            ResidualBottleneck(N, N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, 3),
        )
        # 使用identity进行占位
        self.depth_synthesis_transform = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N),
            nn.Identity(),
            ResidualBottleneck(2 * N, N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            AttentionBlock(N),
            nn.Identity(),
            ResidualBottleneck(2 * N, N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            nn.Identity(),
            ResidualBottleneck(2 * N, N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, 1),
        )

    def forward(self, rgb, depth):
        rgb_hat = rgb
        depth_hat = depth
        for num, (rgb_bk, depth_bk) in enumerate(zip(self.rgb_synthesis_transform, self.depth_synthesis_transform)):
            if isinstance(rgb_bk, bi_spf_single):
                depth_f = rgb_bk(rgb_hat, depth_hat)
                depth_hat = torch.cat((depth_hat, depth_f), dim=-3)
            else:
                rgb_hat = rgb_bk(rgb_hat)
                depth_hat = depth_bk(depth_hat)

        return rgb_hat, depth_hat


###################### HyperSynthesisEX ######################


class HyperSynthesis(nn.Module):
    """
    Local Reference
    """

    def __init__(self, M=192, N=192) -> None:
        super().__init__()
        self.M = M
        self.N = N

        self.increase = nn.Sequential(
            conv3x3(N, M),
            nn.GELU(),
            subpel_conv3x3(M, M, 2),
            nn.GELU(),
            conv3x3(M, M * 3 // 2),
            nn.GELU(),
            subpel_conv3x3(M * 3 // 2, M * 3 // 2, 2),
            nn.GELU(),
            conv3x3(M * 3 // 2, M * 2),
        )

    def forward(self, x):
        x = self.increase(x)

        return x


class HyperSynthesisEX(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.increase = nn.Sequential(
            deconv(N, M), act(), deconv(M, M * 3 // 2), act(), deconv(M * 3 // 2, M * 2, kernel_size=3, stride=1)
        )

    def forward(self, x):
        x = self.increase(x)
        return x


# 接受rgb和depth作为输入输出，不需要进行交互
class HyperSynthesisEXcro(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.rgb_increase = nn.Sequential(
            deconv(N, M), act(), deconv(M, M * 3 // 2), act(), deconv(M * 3 // 2, M * 2, kernel_size=3, stride=1)
        )
        self.depth_increase = nn.Sequential(
            deconv(N, M), act(), deconv(M, M * 3 // 2), act(), deconv(M * 3 // 2, M * 2, kernel_size=3, stride=1)
        )

    def forward(self, rgb, depth):
        rgb = self.rgb_increase(rgb)
        depth = self.depth_increase(depth)
        return rgb, depth


class HyperSynthesisEXcross(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.r_h_s1 = hyper_transform_block(2 * N, M)
        self.r_h_s2 = hyper_transform_block(2 * M, M * 3 // 2)
        self.r_h_s3 = hyper_transform_block(M * 3, 2 * M, True)

        self.d_h_s1 = hyper_transform_block(2 * N, M)
        self.d_h_s2 = hyper_transform_block(2 * M, M * 3 // 2)
        self.d_h_s3 = hyper_transform_block(M * 3, 2 * M, True)

    def forward(self, rgb, depth):
        r1 = self.r_h_s1(rgb, depth)
        d1 = self.d_h_s1(depth, rgb)
        r2 = self.r_h_s2(r1, d1)
        d2 = self.d_h_s2(d1, r1)
        r_params = self.r_h_s3(r2, d2)
        d_params = self.d_h_s3(d2, r2)
        return r_params, d_params

class HyperSynthesisEXSingle(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.r_h_s1 = hyper_transform_block_single( N, M)
        self.r_h_s2 = hyper_transform_block_single(M, M * 3 // 2)
        self.r_h_s3 = hyper_transform_block_single(M * 3//2, 2 * M, True)

        self.d_h_s1 = hyper_transform_block(2 * N, M)
        self.d_h_s2 = hyper_transform_block(2 * M, M * 3 // 2)
        self.d_h_s3 = hyper_transform_block(M * 3, 2 * M, True)

    def forward(self, rgb, depth):
        r1 = self.r_h_s1(rgb)
        d1 = self.d_h_s1(depth, rgb)
        r2 = self.r_h_s2(r1)
        d2 = self.d_h_s2(d1, r1)
        r_params = self.r_h_s3(r2)
        d_params = self.d_h_s3(d2, r2)
        return r_params, d_params

class hyper_transform_block(nn.Module):
    def __init__(self, in_channel, out_channel, is_last=False):
        super(hyper_transform_block, self).__init__()
        self.se = SE_Block(in_channel)
        if is_last is False:
            self.relu = nn.LeakyReLU(inplace=True)
            self.deconv = deconv(in_channel, out_channel, stride=2, kernel_size=5)
        else:
            self.relu = None
            self.deconv = deconv(in_channel, out_channel, stride=1, kernel_size=3)  

    def forward(self, rgb, depth):
        f = torch.cat((rgb, depth), dim=-3)
        f = self.se(f)
        f = self.deconv(f)
        if self.relu is not None:
            f = self.relu(f)
        return f

class hyper_transform_block_single(nn.Module):
    def __init__(self, in_channel, out_channel, is_last=False):
        super(hyper_transform_block_single, self).__init__()
        self.se = SE_Block(in_channel)
        if is_last is False:
            self.relu = nn.LeakyReLU(inplace=True)
            self.deconv = deconv(in_channel, out_channel, stride=2, kernel_size=5)
        else:
            self.relu = None
            self.deconv = deconv(in_channel, out_channel, stride=1, kernel_size=3) 

    def forward(self, f):
        f = self.se(f)
        f = self.deconv(f)
        if self.relu is not None:
            f = self.relu(f)
        return f
        

