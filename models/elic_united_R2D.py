import torch
import torch.nn as nn
from modules.transform import *
from utils.ckbd import *

from .elic_united import ELIC_united


class ELIC_united_R2D(ELIC_united):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        N = config.N
        M = config.M
        slice_num = config.slice_num
        slice_ch = config.slice_ch
        self.quant = config.quant  # noise or ste
        self.slice_num = slice_num
        self.slice_ch = slice_ch
        self.g_a = AnalysisTransformEXSingle(N, M, act=nn.ReLU)
        self.g_s = SynthesisTransformEXSingle(N, M, act=nn.ReLU)
        self.h_a = HyperAnalysisEXcross(N, M, act=nn.ReLU)
        self.h_s = HyperSynthesisEXSingle(N, M, act=nn.ReLU)

        self.rgb_local_context = nn.ModuleList(
            nn.Conv2d(in_channels=slice_ch[i], out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2)
            for i in range(len(slice_ch))
        )
        self.rgb_local_context_anchor_with_nonanchor = nn.ModuleList(
            nn.Conv2d(in_channels=slice_ch[i], out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2)
            for i in range(len(slice_ch))
        )
        self.depth_local_context = nn.ModuleList(
            nn.Conv2d(in_channels=slice_ch[i], out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2)
            for i in range(len(slice_ch))
        )

        self.rgb_channel_context = nn.ModuleList(
            ChannelContextEX(in_dim=sum(slice_ch[:i]), out_dim=slice_ch[i] * 2, act=nn.ReLU) if i else None
            for i in range(slice_num)
        )
        self.depth_channel_context = nn.ModuleList(
            ChannelContextEX(in_dim=sum(slice_ch[:i]), out_dim=slice_ch[i] * 2, act=nn.ReLU) if i else None
            for i in range(slice_num)
        )

        self.rgb_entropy_parameters_anchor = nn.ModuleList(
            EntropyParametersEX(in_dim=(M * 2 + slice_ch[i] * 2), out_dim=slice_ch[i] * 2, act=nn.ReLU)
            if i
            else EntropyParametersEX(in_dim=M * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )
        self.depth_entropy_parameters_anchor = nn.ModuleList(
            EntropyParametersEX(in_dim=(M * 4 + slice_ch[i] * 6), out_dim=slice_ch[i] * 2, act=nn.ReLU)
            if i
            else EntropyParametersEX(in_dim=M * 4 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )

        self.rgb_entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 4, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            if i
            else EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )
        self.depth_entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 4 + slice_ch[i] * 4 + slice_ch[i] * 4, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            if i
            else EntropyParametersEX(in_dim=M * 4 + slice_ch[i] * 4, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )

    def entropy_estimate_one_slice(
        self,
        rgb_y_slice,
        depth_y_slice,
        rgb_hyper_params,
        depth_hyper_params,
        rgb_y_hat_slices,
        depth_y_hat_slices,
        idx,
    ):
        rgb_slice_anchor, rgb_slice_nonanchor = ckbd_split(rgb_y_slice)
        depth_slice_anchor, depth_slice_nonanchor = ckbd_split(depth_y_slice)
        rgb_init_context = [rgb_hyper_params]
        depth_init_context = [rgb_hyper_params, depth_hyper_params]

        if idx != 0:
            rgb_channel_ctx = self.rgb_channel_context[idx](torch.cat(rgb_y_hat_slices, dim=1))
            depth_channel_ctx = self.depth_channel_context[idx](torch.cat(depth_y_hat_slices, dim=1))
            rgb_init_context = [rgb_hyper_params, rgb_channel_ctx]
            depth_init_context = [rgb_hyper_params, depth_hyper_params, rgb_channel_ctx, depth_channel_ctx]

        ## anchor-part coding
        rgb_slice_anchor, rgb_scales_anchor, rgb_means_anchor, rgb_local_ctx = self.codeOnePart(
            rgb_slice_anchor,
            rgb_init_context,
            self.rgb_entropy_parameters_anchor[idx],
            ckbd_anchor,
            self.rgb_gaussian_conditional,
            local_context=self.rgb_local_context[idx],
        )
        depth_slice_anchor, depth_scales_anchor, depth_means_anchor, depth_local_ctx = self.codeOnePart(
            depth_slice_anchor,
            [rgb_local_ctx] + depth_init_context,
            self.depth_entropy_parameters_anchor[idx],
            ckbd_anchor,
            self.depth_gaussian_conditional,
            local_context=self.depth_local_context[idx],
        )

        ## nonanchor-part coding
        (
            rgb_slice_nonanchor,
            rgb_scales_nonanchor,
            rgb_means_nonanchor,
            rgb_local_ctx,
            rgb_y_hat_slice,
        ) = self.codeOnePart(
            rgb_slice_nonanchor,
            [rgb_local_ctx] + rgb_init_context,
            self.rgb_entropy_parameters_nonanchor[idx],
            ckbd_nonanchor,
            self.rgb_gaussian_conditional,
            anchor_part=rgb_slice_anchor,
            local_context=self.rgb_local_context_anchor_with_nonanchor[idx],
        )
        depth_slice_nonanchor, depth_scales_nonanchor, depth_means_nonanchor, depth_y_hat_slice = self.codeOnePart(
            depth_slice_nonanchor,
            [rgb_local_ctx, depth_local_ctx] + depth_init_context,
            self.depth_entropy_parameters_nonanchor[idx],
            ckbd_nonanchor,
            self.depth_gaussian_conditional,
            anchor_part=depth_slice_anchor,
        )

        ## bpp estimate
        rgb_scales_slice = ckbd_merge(rgb_scales_anchor, rgb_scales_nonanchor)
        rgb_means_slice = ckbd_merge(rgb_means_anchor, rgb_means_nonanchor)
        _, rgb_y_slice_likelihoods = self.rgb_gaussian_conditional(rgb_y_slice, rgb_scales_slice, rgb_means_slice)

        depth_scales_slice = ckbd_merge(depth_scales_anchor, depth_scales_nonanchor)
        depth_means_slice = ckbd_merge(depth_means_anchor, depth_means_nonanchor)
        _, depth_y_slice_likelihoods = self.depth_gaussian_conditional(
            depth_y_slice, depth_scales_slice, depth_means_slice
        )

        return rgb_y_hat_slice, depth_y_hat_slice, rgb_y_slice_likelihoods, depth_y_slice_likelihoods

    def compress_one_slice(
        self,
        rgb_y_slice,
        depth_y_slice,
        rgb_hyper_params,
        depth_hyper_params,
        rgb_y_hat_slices,
        depth_y_hat_slices,
        idx,
        rgb_symbols_list,
        rgb_indexes_list,
        depth_symbols_list,
        depth_indexes_list,
    ):
        rgb_slice_anchor, rgb_slice_nonanchor = ckbd_split(rgb_y_slice)
        depth_slice_anchor, depth_slice_nonanchor = ckbd_split(depth_y_slice)
        rgb_init_context = [rgb_hyper_params]
        depth_init_context = [rgb_hyper_params, depth_hyper_params]
        if idx != 0:
            rgb_channel_ctx = self.rgb_channel_context[idx](torch.cat(rgb_y_hat_slices, dim=1))
            depth_channel_ctx = self.depth_channel_context[idx](torch.cat(depth_y_hat_slices, dim=1))
            rgb_init_context = [rgb_hyper_params, rgb_channel_ctx]
            depth_init_context = [rgb_hyper_params, depth_hyper_params, rgb_channel_ctx, depth_channel_ctx]

        # rgb-anchor
        rgb_params_anchor = self.rgb_entropy_parameters_anchor[idx](torch.cat(rgb_init_context, dim=1))
        rgb_scales_anchor, rgb_means_anchor = rgb_params_anchor.chunk(2, 1)
        rgb_slice_anchor = compress_anchor(
            self.rgb_gaussian_conditional,
            rgb_slice_anchor,
            rgb_scales_anchor,
            rgb_means_anchor,
            rgb_symbols_list,
            rgb_indexes_list,
        )
        rgb_local_ctx = self.rgb_local_context[idx](rgb_slice_anchor)

        # depth-anchor
        depth_params_anchor = self.depth_entropy_parameters_anchor[idx](
            torch.cat([rgb_local_ctx] + depth_init_context, dim=1)
        )
        depth_scales_anchor, depth_means_anchor = depth_params_anchor.chunk(2, 1)
        depth_slice_anchor = compress_anchor(
            self.depth_gaussian_conditional,
            depth_slice_anchor,
            depth_scales_anchor,
            depth_means_anchor,
            depth_symbols_list,
            depth_indexes_list,
        )
        depth_local_ctx = self.depth_local_context[idx](depth_slice_anchor)

        # rgb-nonanchor
        rgb_params_nonanchor = self.rgb_entropy_parameters_nonanchor[idx](
            torch.cat([rgb_local_ctx] + rgb_init_context, dim=1)
        )
        rgb_scales_nonanchor, rgb_means_nonanchor = rgb_params_nonanchor.chunk(2, 1)
        rgb_slice_nonanchor = compress_nonanchor(
            self.rgb_gaussian_conditional,
            rgb_slice_nonanchor,
            rgb_scales_nonanchor,
            rgb_means_nonanchor,
            rgb_symbols_list,
            rgb_indexes_list,
        )
        rgb_y_hat_slice = rgb_slice_nonanchor + rgb_slice_anchor
        rgb_local_ctx = self.rgb_local_context_anchor_with_nonanchor[idx](rgb_y_hat_slice)

        # depth-nonanchor
        depth_params_nonanchor = self.depth_entropy_parameters_nonanchor[idx](
            torch.cat([rgb_local_ctx, depth_local_ctx] + depth_init_context, dim=1)
        )
        depth_scales_nonanchor, depth_means_nonanchor = depth_params_nonanchor.chunk(2, 1)
        depth_slice_nonanchor = compress_nonanchor(
            self.depth_gaussian_conditional,
            depth_slice_nonanchor,
            depth_scales_nonanchor,
            depth_means_nonanchor,
            depth_symbols_list,
            depth_indexes_list,
        )
        depth_y_hat_slice = depth_slice_nonanchor + depth_slice_anchor

        rgb_y_hat_slices.append(rgb_y_hat_slice)
        depth_y_hat_slices.append(depth_y_hat_slice)
        return rgb_y_hat_slices, depth_y_hat_slices

    def decompress_one_slice(
        self,
        rgb_decoder,
        depth_decoder,
        rgb_y_hat_slices,
        depth_y_hat_slices,
        rgb_hyper_params,
        depth_hyper_params,
        idx,
        rgb_cdf,
        rgb_cdf_lengths,
        rgb_offsets,
        depth_cdf,
        depth_cdf_lengths,
        depth_offsets,
    ):
        rgb_init_context = [rgb_hyper_params]
        depth_init_context = [rgb_hyper_params, depth_hyper_params]
        if idx != 0:
            rgb_channel_ctx = self.rgb_channel_context[idx](torch.cat(rgb_y_hat_slices, dim=1))
            depth_channel_ctx = self.depth_channel_context[idx](torch.cat(depth_y_hat_slices, dim=1))
            rgb_init_context = [rgb_hyper_params, rgb_channel_ctx]
            depth_init_context = [rgb_hyper_params, depth_hyper_params, rgb_channel_ctx, depth_channel_ctx]

        # rgb-anchor
        rgb_params_anchor = self.rgb_entropy_parameters_anchor[idx](torch.cat(rgb_init_context, dim=1))
        rgb_scales_anchor, rgb_means_anchor = rgb_params_anchor.chunk(2, 1)
        rgb_slice_anchor = decompress_anchor(
            self.rgb_gaussian_conditional,
            rgb_scales_anchor,
            rgb_means_anchor,
            rgb_decoder,
            rgb_cdf,
            rgb_cdf_lengths,
            rgb_offsets,
        )
        rgb_local_ctx = self.rgb_local_context[idx](rgb_slice_anchor)

        # depth-anchor
        depth_params_anchor = self.depth_entropy_parameters_anchor[idx](
            torch.cat([rgb_local_ctx] + depth_init_context, dim=1)
        )
        depth_scales_anchor, depth_means_anchor = depth_params_anchor.chunk(2, 1)
        depth_slice_anchor = decompress_anchor(
            self.depth_gaussian_conditional,
            depth_scales_anchor,
            depth_means_anchor,
            depth_decoder,
            depth_cdf,
            depth_cdf_lengths,
            depth_offsets,
        )
        depth_local_ctx = self.depth_local_context[idx](depth_slice_anchor)
        rgb_params_nonanchor = self.rgb_entropy_parameters_nonanchor[idx](
            torch.cat([rgb_local_ctx] + rgb_init_context, dim=1)
        )

        # rgb-nonanchor
        rgb_scales_nonanchor, rgb_means_nonanchor = rgb_params_nonanchor.chunk(2, 1)
        rgb_slice_nonanchor = decompress_nonanchor(
            self.rgb_gaussian_conditional,
            rgb_scales_nonanchor,
            rgb_means_nonanchor,
            rgb_decoder,
            rgb_cdf,
            rgb_cdf_lengths,
            rgb_offsets,
        )
        rgb_y_hat_slice = rgb_slice_nonanchor + rgb_slice_anchor
        rgb_y_hat_slices.append(rgb_y_hat_slice)
        rgb_local_ctx = self.rgb_local_context_anchor_with_nonanchor[idx](rgb_y_hat_slice)

        # depth-nonanchor
        depth_params_nonanchor = self.depth_entropy_parameters_nonanchor[idx](
            torch.cat([rgb_local_ctx, depth_local_ctx] + depth_init_context, dim=1)
        )
        depth_scales_nonanchor, depth_means_nonanchor = depth_params_nonanchor.chunk(2, 1)
        depth_slice_nonanchor = decompress_nonanchor(
            self.depth_gaussian_conditional,
            depth_scales_nonanchor,
            depth_means_nonanchor,
            depth_decoder,
            depth_cdf,
            depth_cdf_lengths,
            depth_offsets,
        )
        depth_y_hat_slice = depth_slice_nonanchor + depth_slice_anchor
        depth_y_hat_slices.append(depth_y_hat_slice)

        return rgb_y_hat_slices, depth_y_hat_slices
