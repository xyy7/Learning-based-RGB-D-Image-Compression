import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.layers.conv import conv, conv1x1, conv3x3, deconv
from modules.layers.res_blk import *
from modules.transform.attention import bi_spf
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .elic_united import ELIC_united


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
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


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

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

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
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
        inverse=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
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
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchSplit(nn.Module):
    """Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(dim, dim * 2, bias=False)
        self.norm = norm_layer(dim)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.norm(x)
        x = self.reduction(x)  # B, L, C
        x = x.permute(0, 2, 1).contiguous().view(B, 2 * C, H, W)
        x = self.shuffle(x)  # B, C//2 ,2H, 2W
        x = x.permute(0, 2, 3, 1).contiguous().view(B, 4 * L, -1)
        return x


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        inverse=False,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    inverse=inverse,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
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

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            if isinstance(self.downsample, PatchMerging):
                Wh, Ww = (H + 1) // 2, (W + 1) // 2
            elif isinstance(self.downsample, PatchSplit):
                Wh, Ww = H * 2, W * 2
            return x_down, Wh, Ww
        else:
            return x, H, W


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class AnalysisTransformSTFunited(nn.Module):
    def __init__(
        self,
        pretrain_img_size=256,
        patch_size=2,
        embed_dim=48,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        **kargs,
    ):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        self.rgb_patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=3, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None
        )

        self.depth_patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=1, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.rgb_ana_layers = nn.ModuleList()
        self.depth_ana_layers = nn.ModuleList()
        dim = embed_dim
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                inverse=False,
            )
            dim *= 2
            self.rgb_ana_layers.append(layer)
            self.depth_ana_layers.append(copy.deepcopy(layer))
            if i_layer < self.num_layers - 1:
                self.rgb_ana_layers.append(bi_spf(dim))
                # self.rgb_ana_layers.append(conv1x1(2 * dim, dim, stride=1))
                self.depth_ana_layers.append(nn.Identity())
                # self.depth_ana_layers.append(conv1x1(2 * dim, dim, stride=1))

    def forward(self, rgb, depth):
        rgb = self.rgb_patch_embed(rgb)
        depth = self.depth_patch_embed(depth)
        B, C, Wh, Ww = rgb.shape

        rgb = rgb.flatten(2).transpose(1, 2)
        depth = depth.flatten(2).transpose(1, 2)
        rgb = self.pos_drop(rgb)
        depth = self.pos_drop(depth)

        rgb_y = rgb
        depth_y = depth

        for num, (rgb_bk, depth_bk) in enumerate(zip(self.rgb_ana_layers, self.depth_ana_layers)):
            if isinstance(rgb_bk, bi_spf):
                rgb_y = rgb_y.view(B, Wh, Ww, -1).permute(0, 3, 1, 2).contiguous()
                depth_y = depth_y.view(B, Wh, Ww, -1).permute(0, 3, 1, 2).contiguous()
                depth_y = depth_bk(depth_y)  # nn.Identity
                rgb_f, depth_f = rgb_bk(rgb_y, depth_y)  # ESA attention
                rgb_y = rgb_y + rgb_f
                depth_y = depth_y + depth_f
            #     rgb_y = torch.cat((rgb_y, rgb_f), dim=-3)
            #     depth_y = torch.cat((depth_y, depth_f), dim=-3)
            # elif isinstance(rgb_bk, nn.Conv2d):
            #     rgb_y = rgb_bk(rgb_y)  # conv5x5
            #     depth_y = depth_bk(depth_y)  # conv5x5
                rgb_y = rgb_y.flatten(2).transpose(1, 2)
                depth_y = depth_y.flatten(2).transpose(1, 2)
            else:
                rgb_y, _, _ = rgb_bk(rgb_y, Wh, Ww)  # swin-transformer block
                depth_y, Wh, Ww = depth_bk(depth_y, Wh, Ww)  # swin-transformer block

        C = self.embed_dim * 8
        rgb_y = rgb_y.view(-1, Wh, Ww, C).permute(0, 3, 1, 2).contiguous()
        depth_y = depth_y.view(-1, Wh, Ww, C).permute(0, 3, 1, 2).contiguous()
        return rgb_y, depth_y


class SynthesisTransformSTFunited(nn.Module):
    def __init__(
        self,
        pretrain_img_size=256,
        patch_size=2,
        embed_dim=48,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        **kargs,
    ) -> None:
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        depths = depths[::-1]
        num_heads = num_heads[::-1]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.rgb_syn_layers = nn.ModuleList()
        self.depth_syn_layers = nn.ModuleList()
        dim = embed_dim * 8
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchSplit if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                inverse=True,
            )
            dim = dim // 2
            self.rgb_syn_layers.append(layer)
            self.depth_syn_layers.append(copy.deepcopy(layer))
            if i_layer < self.num_layers - 1:
                self.rgb_syn_layers.append(bi_spf(dim))
                # self.rgb_syn_layers.append(conv1x1(2 * dim, dim, stride=1))
                self.depth_syn_layers.append(nn.Identity())
                # self.depth_syn_layers.append(conv1x1(2 * dim, dim, stride=1))

        self.rgb_end_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * patch_size**2, kernel_size=5, stride=1, padding=2),
            nn.PixelShuffle(patch_size),
            nn.Conv2d(embed_dim, 3, kernel_size=3, stride=1, padding=1),
        )
        self.depth_end_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * patch_size**2, kernel_size=5, stride=1, padding=2),
            nn.PixelShuffle(patch_size),
            nn.Conv2d(embed_dim, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, rgb, depth):
        B, C, Wh, Ww = rgb.shape
        rgb_hat = rgb
        depth_hat = depth
        rgb_hat = rgb_hat.permute(0, 2, 3, 1).contiguous().view(-1, Wh * Ww, C)
        depth_hat = depth_hat.permute(0, 2, 3, 1).contiguous().view(-1, Wh * Ww, C)

        for num, (rgb_bk, depth_bk) in enumerate(zip(self.rgb_syn_layers, self.depth_syn_layers)):
            if isinstance(rgb_bk, bi_spf):
                rgb_hat = rgb_hat.view(B, Wh, Ww, -1).permute(0, 3, 1, 2).contiguous()
                depth_hat = depth_hat.view(B, Wh, Ww, -1).permute(0, 3, 1, 2).contiguous()
                depth_hat = depth_bk(depth_hat)
                rgb_f, depth_f = rgb_bk(rgb_hat, depth_hat)
                rgb_hat = rgb_hat + rgb_f
                depth_hat = depth_hat + depth_f
                # rgb_hat = torch.cat((rgb_hat, rgb_f), dim=-3)
                # depth_hat = torch.cat((depth_hat, depth_f), dim=-3)
                # elif isinstance(rgb_bk, nn.Conv2d):
                #     rgb_hat = rgb_bk(rgb_hat)
                #     depth_hat = depth_bk(depth_hat)
                rgb_hat = rgb_hat.flatten(2).transpose(1, 2)
                depth_hat = depth_hat.flatten(2).transpose(1, 2)
            else:
                rgb_hat, _, _ = rgb_bk(rgb_hat, Wh, Ww)
                depth_hat, Wh, Ww = depth_bk(depth_hat, Wh, Ww)

        rgb_hat = self.rgb_end_conv(rgb_hat.view(-1, Wh, Ww, self.embed_dim).permute(0, 3, 1, 2).contiguous())
        depth_hat = self.depth_end_conv(depth_hat.view(-1, Wh, Ww, self.embed_dim).permute(0, 3, 1, 2).contiguous())

        return rgb_hat, depth_hat


class SymmetricalTransFormerUnited(ELIC_united):
    def __init__(
        self,
        pretrain_img_size=256,
        patch_size=2,
        channel=4,
        embed_dim=48,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=4,
        num_slices=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        config=None,
        **kwargs,
    ):
        config.slice_ch = [24, 24, 48, 96, 192]
        config.N = 192
        config.M = 384
        super().__init__(config=config, **kwargs)
        self.g_a = AnalysisTransformSTFunited(
            pretrain_img_size=pretrain_img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            num_slices=num_slices,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
        )
        self.g_s = SynthesisTransformSTFunited(
            pretrain_img_size=pretrain_img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            num_slices=num_slices,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
        )
