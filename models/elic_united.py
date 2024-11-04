import time

import torch
import torch.nn as nn
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel
from compressai.ops import ste_round
from modules.transform import *
from utils.ckbd import *
from utils.moduleFunc import get_scale_table, update_registered_buffers


class ELIC_united(CompressionModel):
    def __init__(self, config, **kwargs):
        super().__init__(config.N, **kwargs)

        N = config.N
        M = config.M
        slice_num = config.slice_num
        slice_ch = config.slice_ch
        self.quant = config.quant  # noise or ste
        self.slice_num = slice_num
        self.slice_ch = slice_ch
        self.g_a = AnalysisTransformEXcross(N, M, act=nn.ReLU)
        self.g_s = SynthesisTransformEXcross(N, M, act=nn.ReLU)
        self.h_a = HyperAnalysisEXcross(N, M, act=nn.ReLU)
        self.h_s = HyperSynthesisEXcross(N, M, act=nn.ReLU)

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

        # 因为是使用索引idx来跳过None，第一个idx，没有使用channel信息
        self.rgb_channel_context = nn.ModuleList(
            ChannelContextEX(in_dim=sum(slice_ch[:i]), out_dim=slice_ch[i] * 2, act=nn.ReLU) if i else None
            for i in range(slice_num)
        )
        self.depth_channel_context = nn.ModuleList(
            ChannelContextEX(in_dim=sum(slice_ch[:i]), out_dim=slice_ch[i] * 2, act=nn.ReLU) if i else None
            for i in range(slice_num)
        )

        self.rgb_entropy_parameters_anchor = nn.ModuleList(
            EntropyParametersEX(in_dim=(M * 4 + slice_ch[i] * 4), out_dim=slice_ch[i] * 2, act=nn.ReLU)
            if i
            else EntropyParametersEX(in_dim=M * 4, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )
        self.depth_entropy_parameters_anchor = nn.ModuleList(
            EntropyParametersEX(in_dim=(M * 4 + slice_ch[i] * 6), out_dim=slice_ch[i] * 2, act=nn.ReLU)
            if i
            else EntropyParametersEX(in_dim=M * 4 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )

        # Use channel_ctx and hyper_params 【+ cross modal channel ctx + cross modal hyper_params + cross modal local for rgb + depth】
        self.rgb_entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 4 + slice_ch[i] * 4 + slice_ch[i] * 4, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            if i
            else EntropyParametersEX(in_dim=M * 4 + slice_ch[i] * 4, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )
        self.depth_entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 4 + slice_ch[i] * 4 + slice_ch[i] * 4, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            if i
            else EntropyParametersEX(in_dim=M * 4 + slice_ch[i] * 4, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )

        self.entropy_bottleneck = None
        self.rgb_entropy_bottleneck = EntropyBottleneck(N)
        self.depth_entropy_bottleneck = EntropyBottleneck(N)

        self.gaussianConditional = None
        self.rgb_gaussian_conditional = GaussianConditional(None)
        self.depth_gaussian_conditional = GaussianConditional(None)

    def count_parameters(self, only_trainable=False):
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def codeOnePart(self, slice_part, contextList, params_module, split_func, entropy_model, **kargs):
        params_one_part = params_module(torch.cat(contextList, 1))
        scales, means = params_one_part.chunk(2, 1)
        scales = split_func(scales)
        means = split_func(means)
        if self.quant == "ste":
            slice_part = ste_round(slice_part - means) + means
        else:
            slice_part = entropy_model.quantize(slice_part, "noise" if self.training else "dequantize")
            slice_part = split_func(slice_part)

        if "anchor_part" in kargs.keys():
            slices = slice_part + kargs["anchor_part"]
            if "local_context" in kargs.keys():
                localctx = kargs["local_context"](slices)
                return slice_part, scales, means, localctx, slices
            return slice_part, scales, means, slices

        if "local_context" in kargs.keys():
            localctx = kargs["local_context"](slice_part)
            return slice_part, scales, means, localctx
        return slice_part, scales, means

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
        init_context = [rgb_hyper_params, depth_hyper_params]

        if idx != 0:
            rgb_channel_ctx = self.rgb_channel_context[idx](torch.cat(rgb_y_hat_slices, dim=1))
            depth_channel_ctx = self.depth_channel_context[idx](torch.cat(depth_y_hat_slices, dim=1))
            init_context = [rgb_hyper_params, depth_hyper_params, rgb_channel_ctx, depth_channel_ctx]

        ## anchor-part coding
        rgb_slice_anchor, rgb_scales_anchor, rgb_means_anchor, rgb_local_ctx = self.codeOnePart(
            rgb_slice_anchor,
            init_context,
            self.rgb_entropy_parameters_anchor[idx],
            ckbd_anchor,
            self.rgb_gaussian_conditional,
            local_context=self.rgb_local_context[idx],
        )
        depth_slice_anchor, depth_scales_anchor, depth_means_anchor, depth_local_ctx = self.codeOnePart(
            depth_slice_anchor,
            [rgb_local_ctx] + init_context,
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
            [rgb_local_ctx, depth_local_ctx] + init_context,
            self.rgb_entropy_parameters_nonanchor[idx],
            ckbd_nonanchor,
            self.rgb_gaussian_conditional,
            anchor_part=rgb_slice_anchor,
            local_context=self.rgb_local_context_anchor_with_nonanchor[idx],
        )
        depth_slice_nonanchor, depth_scales_nonanchor, depth_means_nonanchor, depth_y_hat_slice = self.codeOnePart(
            depth_slice_nonanchor,
            [rgb_local_ctx, depth_local_ctx] + init_context,
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

    def entropy_estimate_united(self, rgb, depth, rgb_hyper_params, depth_hyper_params):
        rgb_y_slices = [
            rgb[:, sum(self.slice_ch[:i]) : sum(self.slice_ch[: (i + 1)]), ...] for i in range(len(self.slice_ch))
        ]
        rgb_y_hat_slices = []
        rgb_y_likelihoods = []

        depth_y_slices = [
            depth[:, sum(self.slice_ch[:i]) : sum(self.slice_ch[: (i + 1)]), ...] for i in range(len(self.slice_ch))
        ]
        depth_y_hat_slices = []
        depth_y_likelihoods = []

        for idx, (rgb_y_slice, depth_y_slice) in enumerate(zip(rgb_y_slices, depth_y_slices)):
            (
                rgb_y_hat_slice,
                depth_y_hat_slice,
                rgb_y_slice_likelihoods,
                depth_y_slice_likelihoods,
            ) = self.entropy_estimate_one_slice(
                rgb_y_slice,
                depth_y_slice,
                rgb_hyper_params,
                depth_hyper_params,
                rgb_y_hat_slices,
                depth_y_hat_slices,
                idx,
            )

            rgb_y_hat_slices.append(rgb_y_hat_slice)
            rgb_y_likelihoods.append(rgb_y_slice_likelihoods)
            depth_y_hat_slices.append(depth_y_hat_slice)
            depth_y_likelihoods.append(depth_y_slice_likelihoods)

        rgb_y_hat = torch.cat(rgb_y_hat_slices, dim=1)
        rgb_y_likelihoods = torch.cat(rgb_y_likelihoods, dim=1)

        depth_y_hat = torch.cat(depth_y_hat_slices, dim=1)
        depth_y_likelihoods = torch.cat(depth_y_likelihoods, dim=1)

        return rgb_y_hat, rgb_y_likelihoods, depth_y_hat, depth_y_likelihoods

    def forward(self, rgb, depth):
        rgb_y, depth_y = self.g_a(rgb, depth)
        rgb_z, depth_z = self.h_a(rgb_y, depth_y)

        # bits先验估计
        rgb_z_hat, rgb_z_likelihoods = self.rgb_entropy_bottleneck(rgb_z)
        depth_z_hat, depth_z_likelihoods = self.depth_entropy_bottleneck(depth_z)
        if self.quant == "ste":
            rgb_z_offset = self.rgb_entropy_bottleneck._get_medians()
            rgb_z_hat = ste_round(rgb_z - rgb_z_offset) + rgb_z_offset
            depth_z_offset = self.depth_entropy_bottleneck._get_medians()
            depth_z_hat = ste_round(depth_z - depth_z_offset) + depth_z_offset

        # Hyper-parameters
        rgb_hyper_params, depth_hyper_params = self.h_s(rgb_z_hat, depth_z_hat)
        rgb_y_hat, rgb_y_likelihoods, depth_y_hat, depth_y_likelihoods = self.entropy_estimate_united(
            rgb_y, depth_y, rgb_hyper_params, depth_hyper_params
        )

        rgb_hat, depth_hat = self.g_s(rgb_y_hat, depth_y_hat)

        # print("united forward save npy")
        # np.save("R2D_rgb_y_li.npy", rgb_y_likelihoods.cpu().numpy())
        # np.save("R2D_depth_y_li.npy", depth_y_likelihoods.cpu().numpy())

        return {
            "x_hat": {"r": rgb_hat, "d": depth_hat},
            "r_likelihoods": {"y": rgb_y_likelihoods, "z": rgb_z_likelihoods},
            "d_likelihoods": {"y": depth_y_likelihoods, "z": depth_z_likelihoods},
        }

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
        init_context = [rgb_hyper_params, depth_hyper_params]
        if idx != 0:
            rgb_channel_ctx = self.rgb_channel_context[idx](torch.cat(rgb_y_hat_slices, dim=1))
            depth_channel_ctx = self.depth_channel_context[idx](torch.cat(depth_y_hat_slices, dim=1))
            init_context = [rgb_hyper_params, depth_hyper_params, rgb_channel_ctx, depth_channel_ctx]

        # rgb-anchor
        rgb_params_anchor = self.rgb_entropy_parameters_anchor[idx](torch.cat(init_context, dim=1))
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
            torch.cat([rgb_local_ctx] + init_context, dim=1)
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
            torch.cat([rgb_local_ctx, depth_local_ctx] + init_context, dim=1)
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
            torch.cat([rgb_local_ctx, depth_local_ctx] + init_context, dim=1)
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

    def compress_united(self, rgb_y, rgb_hyper_params, depth_y, depth_hyper_params):
        rgb_y_slices = [
            rgb_y[:, sum(self.slice_ch[:i]) : sum(self.slice_ch[: (i + 1)]), ...] for i in range(len(self.slice_ch))
        ]
        rgb_y_hat_slices = []

        depth_y_slices = [
            depth_y[:, sum(self.slice_ch[:i]) : sum(self.slice_ch[: (i + 1)]), ...] for i in range(len(self.slice_ch))
        ]
        depth_y_hat_slices = []

        rgb_cdf = self.rgb_gaussian_conditional.quantized_cdf.tolist()
        rgb_cdf_lengths = self.rgb_gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        rgb_offsets = self.rgb_gaussian_conditional.offset.reshape(-1).int().tolist()
        rgb_encoder = BufferedRansEncoder()
        rgb_symbols_list = []
        rgb_indexes_list = []
        rgb_y_strings = []

        depth_cdf = self.depth_gaussian_conditional.quantized_cdf.tolist()
        depth_cdf_lengths = self.depth_gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        depth_offsets = self.depth_gaussian_conditional.offset.reshape(-1).int().tolist()
        depth_encoder = BufferedRansEncoder()
        depth_symbols_list = []
        depth_indexes_list = []
        depth_y_strings = []

        for idx, (rgb_y_slice, depth_y_slice) in enumerate(zip(rgb_y_slices, depth_y_slices)):
            rgb_y_hat_slices, depth_y_hat_slices = self.compress_one_slice(
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
            )

        rgb_encoder.encode_with_indexes(rgb_symbols_list, rgb_indexes_list, rgb_cdf, rgb_cdf_lengths, rgb_offsets)
        rgb_y_string = rgb_encoder.flush()
        rgb_y_strings.append(rgb_y_string)

        depth_encoder.encode_with_indexes(
            depth_symbols_list, depth_indexes_list, depth_cdf, depth_cdf_lengths, depth_offsets
        )
        depth_y_string = depth_encoder.flush()
        depth_y_strings.append(depth_y_string)
        return rgb_y_strings, depth_y_strings

    def compress(self, rgb, depth):
        # print("united compress -> forward and exit()")
        # self.forward(rgb, depth)
        # exit()

        rgb_y, depth_y = self.g_a(rgb, depth)
        rgb_z, depth_z = self.h_a(rgb_y, depth_y)

        # bits先验
        torch.backends.cudnn.deterministic = True
        rgb_z_strings = self.rgb_entropy_bottleneck.compress(rgb_z)
        rgb_z_hat = self.rgb_entropy_bottleneck.decompress(rgb_z_strings, rgb_z.size()[-2:])
        depth_z_strings = self.depth_entropy_bottleneck.compress(depth_z)
        depth_z_hat = self.depth_entropy_bottleneck.decompress(depth_z_strings, depth_z.size()[-2:])

        # Hyper-parameters
        rgb_hyper_params, depth_hyper_params = self.h_s(rgb_z_hat, depth_z_hat)
        rgb_y_strings, depth_y_strings = self.compress_united(rgb_y, rgb_hyper_params, depth_y, depth_hyper_params)

        torch.backends.cudnn.deterministic = False
        return {
            "r_strings": [rgb_y_strings, rgb_z_strings],
            "d_strings": [depth_y_strings, depth_z_strings],
            "shape": rgb_z.size()[-2:],
        }

    def decompress(self, rgb_strings, depth_strings, shape):
        torch.backends.cudnn.deterministic = True
        torch.cuda.synchronize()
        start_time = time.process_time()

        rgb_y_strings = rgb_strings[0][0]  # 本来不需要的，只是compress写成了列表的形式
        rgb_z_strings = rgb_strings[1]
        rgb_z_hat = self.rgb_entropy_bottleneck.decompress(rgb_z_strings, shape)
        depth_y_strings = depth_strings[0][0]
        depth_z_strings = depth_strings[1]
        depth_z_hat = self.depth_entropy_bottleneck.decompress(depth_z_strings, shape)

        rgb_hyper_params, depth_hyper_params = self.h_s(rgb_z_hat, depth_z_hat)

        rgb_y_hat, depth_y_hat = self.decompress_united(
            rgb_y_strings, rgb_hyper_params, depth_y_strings, depth_hyper_params
        )
        torch.backends.cudnn.deterministic = False
        rgb_hat, depth_hat = self.g_s(rgb_y_hat, depth_y_hat)

        torch.cuda.synchronize()
        end_time = time.process_time()
        cost_time = end_time - start_time
        return {"x_hat": {"r": rgb_hat.clamp_(0, 1), "d": depth_hat.clamp_(0, 1)}, "cost_time": cost_time}

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
        init_context = [rgb_hyper_params, depth_hyper_params]
        if idx != 0:
            rgb_channel_ctx = self.rgb_channel_context[idx](torch.cat(rgb_y_hat_slices, dim=1))
            depth_channel_ctx = self.depth_channel_context[idx](torch.cat(depth_y_hat_slices, dim=1))
            init_context = [rgb_hyper_params, depth_hyper_params, rgb_channel_ctx, depth_channel_ctx]

        # rgb-anchor
        rgb_params_anchor = self.rgb_entropy_parameters_anchor[idx](torch.cat(init_context, dim=1))
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
            torch.cat([rgb_local_ctx] + init_context, dim=1)
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
            torch.cat([rgb_local_ctx, depth_local_ctx] + init_context, dim=1)
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
        rgb_local_ctx = self.rgb_local_context_anchor_with_nonanchor[idx](rgb_y_hat_slice)
        rgb_y_hat_slices.append(rgb_y_hat_slice)

        # depth-nonanchor
        depth_params_nonanchor = self.depth_entropy_parameters_nonanchor[idx](
            torch.cat([rgb_local_ctx, depth_local_ctx] + init_context, dim=1)
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

    def decompress_united(self, rgb_y_strings, rgb_hyper_params, depth_y_strings, depth_hyper_params):
        rgb_y_hat_slices = []
        rgb_cdf = self.rgb_gaussian_conditional.quantized_cdf.tolist()
        rgb_cdf_lengths = self.rgb_gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        rgb_offsets = self.rgb_gaussian_conditional.offset.reshape(-1).int().tolist()
        rgb_decoder = RansDecoder()
        rgb_decoder.set_stream(rgb_y_strings)

        depth_y_hat_slices = []
        depth_cdf = self.depth_gaussian_conditional.quantized_cdf.tolist()
        depth_cdf_lengths = self.depth_gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        depth_offsets = self.depth_gaussian_conditional.offset.reshape(-1).int().tolist()
        depth_decoder = RansDecoder()
        depth_decoder.set_stream(depth_y_strings)

        for idx in range(self.slice_num):
            rgb_y_hat_slices, depth_y_hat_slices = self.decompress_one_slice(
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
            )

        rgb_y_hat = torch.cat(rgb_y_hat_slices, dim=1)
        depth_y_hat = torch.cat(depth_y_hat_slices, dim=1)

        return rgb_y_hat, depth_y_hat

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        rgb_updated = self.rgb_gaussian_conditional.update_scale_table(scale_table, force=force)
        depth_updated = self.depth_gaussian_conditional.update_scale_table(scale_table, force=force)
        updated = rgb_updated & depth_updated | super().update(force=force)  # 更新entropybottleneck
        return updated

    def load_state_dict(self, state_dict, strict=False):
        update_registered_buffers(
            self.rgb_gaussian_conditional,
            "rgb_gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.depth_gaussian_conditional,
            "depth_gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.rgb_entropy_bottleneck,
            "rgb_entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.depth_entropy_bottleneck,
            "depth_entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        try:
            super().load_state_dict(state_dict, strict=True)
            print("ELIC_united load state dict strict success.")
        except Exception as e:
            import traceback
            traceback.print_exc(e)
            print("ELIC_united load state dict strict error.")
            super().load_state_dict(state_dict, strict=False)
