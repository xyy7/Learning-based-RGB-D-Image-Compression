import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim


class RateDistortionLossSingleModal(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, quality=1, metrics="mse"):
        super().__init__()
        self.mse = nn.MSELoss()
        lmbdas = [0.00180, 0.00350, 0.00670, 0.01300, 0.02500, 0.04830, 0.09320, 0.1800]
        self.lmbda = lmbdas[int(quality)]
        self.metrics = metrics

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.metrics == "mse":
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["ms_ssim_loss"] = None
            out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]
        elif self.metrics == "ms-ssim":
            out["mse_loss"] = None
            out["ms_ssim_loss"] = 1 - ms_ssim(output["x_hat"], target, data_range=1.0)
            out["loss"] = self.lmbda * out["ms_ssim_loss"] + out["bpp_loss"]

        return out


class RateDistortionLossUnited(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, quality: str, distortionLossForDepth="d_loss", warmup_step=0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1_criterion = nn.L1Loss()
        self.lmbdas = [0.00180, 0.00350, 0.00670, 0.01300, 0.02500, 0.04830, 0.09320, 0.1800]
        self.rgb_lmbda, self.depth_lmbda = self.get_lmbda_from_fraction_q(quality=quality)
        self.distortionLossForDepth = distortionLossForDepth
        self.cur_step = 0
        self.warmup_step = warmup_step

    def get_lmbda_from_fraction_q(self, quality):
        rgb_q, depth_q = quality.split("_")

        def get_lmbda(q):
            q = float(q)
            return (self.lmbdas[math.ceil(q)] + self.lmbdas[math.floor(q)]) / 2

        rgb_lmbda = get_lmbda(rgb_q)
        depth_lmbda = get_lmbda(depth_q)
        return rgb_lmbda, depth_lmbda

    def get_bpp(self, num_pixels, likelihoodsDict):
        return sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in likelihoodsDict.values()
        )

    def get_rgb_loss(self, output, rgb):
        N, _, H, W = rgb.size()
        num_pixels = N * H * W
        loss = {}
        loss["r_bpp_loss"] = self.get_bpp(num_pixels, output["r_likelihoods"])
        r = output["x_hat"]["r"]
        loss["r_mse_loss"] = self.mse(r, rgb)
        loss["rgb_loss"] = self.rgb_lmbda * 255**2 * loss["r_mse_loss"] + loss["r_bpp_loss"]
        return loss

    def get_d_loss(self, d, depth):
        def gradient(x):
            l = x
            r = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
            t = x
            b = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

            dx, dy = torch.abs(r - l), torch.abs(b - t)
            # dx will always have zeros in the last column, r-l
            # dy will always have zeros in the last row,    b-t
            dx[:, :, :, -1] = 0
            dy[:, :, -1, :] = 0

            return dx, dy

        loss = {}
        loss["l1_loss"] = self.l1_criterion(d, depth)
        output_dx, output_dy = gradient(d)
        target_dx, target_dy = gradient(depth)
        grad_diff_x = torch.abs(output_dx - target_dx)
        grad_diff_y = torch.abs(output_dy - target_dy)
        loss["edge_loss"] = torch.mean(grad_diff_x + grad_diff_y)
        loss["ssim_loss"] = torch.clamp((1 - ms_ssim(d, depth, data_range=1)) * 0.5, 0, 1)

        loss["d_loss"] = loss["ssim_loss"] + loss["edge_loss"] + 0.1 * loss["l1_loss"]
        return loss

    def get_depth_loss(self, output, depth):
        N, _, H, W = depth.size()
        num_pixels = N * H * W
        loss = {}
        loss["d_bpp_loss"] = self.get_bpp(num_pixels, output["d_likelihoods"])
        d = output["x_hat"]["d"]

        if self.distortionLossForDepth == "d_loss" and self.cur_step > self.warmup_step:
            loss.update(self.get_d_loss(d, depth))
            loss["depth_loss"] = self.depth_lmbda * 255**2 * 0.01 * loss["d_loss"] + loss["d_bpp_loss"]
            loss["d_mse_loss"] = loss["d_loss"]
        else:
            loss["d_mse_loss"] = self.mse(d, depth)
            loss["d_loss"] = loss["d_mse_loss"]
            loss["depth_loss"] = self.depth_lmbda * 255**2 * loss["d_mse_loss"] + loss["d_bpp_loss"]
        return loss

    def forward(self, output, rgb, depth):
        self.cur_step += 1
        loss = {}
        loss.update(self.get_rgb_loss(output, rgb))
        loss.update(self.get_depth_loss(output, depth))
        loss["loss"] = loss["rgb_loss"] + loss["depth_loss"]
        return loss


if __name__ == "__main__":
    from pprint import pprint

    rgb = torch.rand([8, 3, 480, 640])
    rgb_likelihoods = {
        "y": torch.rand([8, 320, 480 // 8, 640 // 8]) + 0.1,
        "z": torch.rand([8, 192, 480 // 8 // 4, 640 // 8 // 4]) + 0.1,
    }

    depth = torch.rand([8, 1, 480, 640])
    depth_likelihoods = {
        "y": torch.rand([8, 320, 480 // 8, 640 // 8]) + 0.1,
        "z": torch.rand([8, 192, 480 // 8 // 4, 640 // 8 // 4]) + 0.1,
    }

    output_rgb = {"x_hat": rgb, "likelihoods": rgb_likelihoods}
    output_depth = {"x_hat": depth, "likelihoods": depth_likelihoods}
    output_united = {
        "x_hat": {"r": rgb, "d": depth},
        "r_likelihoods": {"y": rgb_likelihoods["y"], "z": rgb_likelihoods["z"]},
        "d_likelihoods": {"y": depth_likelihoods["y"], "z": depth_likelihoods["z"]},
    }

    def test_RateDistortionLossSingleModal():
        loss_single = RateDistortionLossSingleModal()
        loss = loss_single(output_rgb, rgb + rgb)
        pprint(loss)
        print()
        loss = loss_single(output_depth, depth + depth)
        pprint(loss)
        print()

    def test_RateDistortionLossUnited():
        loss_united = RateDistortionLossUnited("2.5_2.5")
        loss = loss_united(output_united, rgb + rgb, depth + depth)
        pprint(loss)
        print()

    test_RateDistortionLossSingleModal()
    test_RateDistortionLossUnited()
