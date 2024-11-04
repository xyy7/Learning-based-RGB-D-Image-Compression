import torch
from torch import nn
from torch.nn import functional as F

from .spatialAligner import Spatial_aligner


def default_conv(in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True, groups=1):
    if not padding and stride == 1:
        padding = kernel_size // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)


class bi_spf_single(nn.Module):
    def __init__(self, N):
        super(bi_spf_single, self).__init__()
        self.r_ext = nn.Conv2d(N, N // 2, stride=1, kernel_size=3, padding=1)
        self.r_act = nn.ReLU()

        self.d_ext = nn.Conv2d(N, N // 2, stride=1, kernel_size=3, padding=1)
        self.d_act = nn.ReLU()
        self.d_esa = ESA(N)

    # 仅仅辅助第二个
    def forward(self, rgb, depth):
        rgb = self.r_ext(rgb)
        rgb = self.r_act(rgb)
        depth = self.d_ext(depth)
        depth = self.d_act(depth)

        d = self.d_esa(torch.cat((depth, rgb), dim=-3))
        return d


class bi_spf(bi_spf_single):
    def __init__(self, N):
        super().__init__(N)
        self.r_esa = ESA(N)

    def forward(self, rgb, depth):
        rgb = self.r_ext(rgb)
        rgb = self.r_act(rgb)
        depth = self.d_ext(depth)
        depth = self.d_act(depth)

        r = self.r_esa(torch.cat((rgb, depth), dim=-3))
        d = self.d_esa(torch.cat((depth, rgb), dim=-3))
        return r, d


# rgb和depth根据空间均值来进行通道加权【通过全局池化来实现】
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道


class ESA(nn.Module):
    def __init__(self, n_feats, conv=default_conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        f = x
        c1_ = self.conv1(f)  # 1*1卷积，降低维度（减少计算复杂度）
        c1 = self.conv2(c1_)  # 减小特征图尺寸
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)  # 减小特征图尺寸，增大感受野
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode="bilinear", align_corners=False)  # 上采样，恢复特征图尺寸
        cf = self.conv_f(c1_)  #
        c4 = self.conv4(c3 + cf)  # 1*1卷积恢复通道数
        m = self.sigmoid(c4)  # 生成mask

        return x * m
