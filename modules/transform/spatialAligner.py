import numpy as np
import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding
    Args:
        img_size (int, int): Image size.  Default: 224,224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=(224, 224), patch_size=2, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_embedding = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 本来这里可以进行归一化层，但是rgbt论文中，是输出embeding的结果，然后再进行layernorm的
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)  # B Ph*Pw C  # flatten(2):2,...,-1 维度进行展平
        if self.norm is not None:
            x = self.norm(x)
        return x


def window_partition(x, window_size=4):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    # in_feature?hidden_feature,out_feature
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)  # 相当于矩阵乘法：[AxB]x[BxC]==>[AxC],输入输出的维度为：BLC 【并不像2D的BCHW】
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5  # or，并不是执行逻辑与操作，而是选择一个非None，非0，非False的数值

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        # 根据窗口大小，设置可学习的位置编码
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv1 = nn.Linear(dim, dim, bias=qkv_bias)  # 线性变换，实际上并不改变来源（不论是否同源）的shape
        self.qkv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)  # qkv1==》query,qkv2==>key,value
        self.proj = nn.Linear(dim, dim)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    # 因为是计算交叉注意力，所以需要传入两个不同的来源的input
    def forward(self, x, guided, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = self.qkv1(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        kv = self.qkv2(guided).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # @符号表示矩阵乘法

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        fused_window_process=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:  # window size基本不可能大于input resolution
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # 隐藏层放大了4倍
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 困难之处：attn mask如何应用在交叉注意力上，需要在意吗？
        # if self.shift_size > 0:
        #     H, W = self.input_resolution
        #     attn_mask = self.create_mask(x, H, W)
        # else:
        #     attn_mask = None
        # self.register_buffer("attn_mask", attn_mask)

        self.fused_window_process = fused_window_process

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, guided, H, W):
        B, L, C = x.shape  # 因为在embedding的时候，已经摊平
        if self.shift_size > 0:
            self.attn_mask = self.create_mask(x, H, W)
        else:
            self.attn_mask = None

        assert L == H * W, "input feature has wrong size"

        shortcut = x

        x = self.norm1(x)
        guided = self.norm1(guided)

        # 转成图像，好进行分块操作
        x = x.view(B, H, W, C)
        guided = guided.view(B, H, W, C)
        # cyclic shift
        # 滑动分块
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

            shifted_guided = torch.roll(guided, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # partition windows
            guided_windows = window_partition(shifted_guided, self.window_size)  # nW*B, window_size, window_size, C

        else:
            # 普通分块
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            guided_windows = window_partition(guided, self.window_size)  # nW*B, window_size, window_size, C

        # 转成BLC，好进行注意力计算
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        guided_windows = guided_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C
        # W-MSA/SW-MSA      窗口多头自注意力/移动窗口多头自注意力 # 普通注意力，attn_mask为0，滑动注意力：
        attn_windows = self.attn(x_windows, guided_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # 恢复成图像，方便进行恢复操作
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x

        # 残差连接
        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))  # drop_path 是不同深度的dropout层

        return x


class Spatial_aligner(nn.Module):
    def __init__(self, in_channel=192, out_channel=192, input_resolution=(224, 224)) -> None:
        super().__init__()
        self.window_size = 4
        self.patch_size = 2
        self.num_head = 3
        self.input_resolution = input_resolution
        self.embed_dim = 96
        self.patch_embeding1 = nn.Conv2d(
            in_channel, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.patch_embeding2 = nn.Conv2d(
            in_channel, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_head,
                    window_size=self.window_size,
                    input_resolution=(
                        self.input_resolution[0] // self.patch_size,
                        self.input_resolution[1] // self.patch_size,
                    ),  # 元组不可以直接除法
                    shift_size=0 if (i % 2 == 0) else 4 // 2,  # 让第二个有shift
                )
                for i in range(2)
            ]
        )  # range(depth) 每块的depth是不同的

        self.recovery = nn.ConvTranspose2d(
            self.embed_dim, out_channel, kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward(self, x, guided):
        B, C, H, W = x.shape
        # 会padding成64，所以肯定是偶数
        embed_x = self.patch_embeding1(x).flatten(2).transpose(1, 2)  # B Ph*Pw C  # flatten(2):2,...,-1 维度进行展平  # BLC
        embed_guided = (
            self.patch_embeding2(guided).flatten(2).transpose(1, 2)
        )  # B Ph*Pw C  # flatten(2):2,...,-1 维度进行展平  # BLC
        for layer in self.blocks:
            embed_x = layer(embed_x, embed_guided, H // self.patch_size, W // self.patch_size)

        out = embed_x.transpose(1, 2)
        # 因为最后一层是MLP，故shape应该为 BLC  # 因为需要进行复原,为了避免padding操作，需要要求图片的shape为偶数
        out = out.contiguous().view(B, self.embed_dim, H // self.patch_size, W // self.patch_size)
        out = self.recovery(out)
        return out
