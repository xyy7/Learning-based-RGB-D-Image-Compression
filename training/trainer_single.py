import os
import sys

sys.path.append("..")  # cd xx/playground

import os

import torch
from utils.IOutils import saveImg
from utils.metrics import AverageMeter, compute_metrics
from utils.rd_loss import RateDistortionLossSingleModal

from .trainer import Trainer


class TrainerSingle(Trainer):
    def __init__(self, args, model_config) -> None:
        super().__init__(args, model_config)
        self.criterion = RateDistortionLossSingleModal(quality=args.quality, metrics=args.metrics)

    def train_one_epoch(self, epoch, current_step, clip_max_norm=1):
        self.net.train()
        for i, d in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            self.aux_optimizer.zero_grad()
            out_criterion, _ = self.forward(d)
            current_step = self.train_backward_and_log(out_criterion, clip_max_norm, i, epoch, current_step)

        return current_step

    def forward(self, d):
        d = d.to(self.device)
        out_net = self.net(d)
        out_criterion = self.criterion(out_net, d)
        return out_criterion, out_net

    def train_backward_and_log(self, out_criterion, clip_max_norm, i, epoch, current_step):
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), clip_max_norm)
        self.optimizer.step()
        aux_loss = self.net.aux_loss()
        aux_loss.backward()
        self.aux_optimizer.step()

        dloss_name = "mse_loss" if out_criterion["mse_loss"] is not None else "ms_ssim_loss"
        current_step += 1
        if (current_step % 100 or self.debug) == 0 and self.rank == 0:
            self.tb_logger.add_scalar("{}".format("[train]: loss"), out_criterion["loss"].item(), current_step)
            self.tb_logger.add_scalar("{}".format("[train]: bpp_loss"), out_criterion["bpp_loss"].item(), current_step)
            self.tb_logger.add_scalar(
                "{}".format(f"[train]: {dloss_name}"), out_criterion[dloss_name].item(), current_step
            )
        if (i % 100 == 0 or self.debug) and self.rank == 0:
            self.logger_train.info(
                f"Train epoch {epoch}: ["
                f"{i*self.train_dataloader.batch_size:5d}/{len(self.train_dataloader.dataset)}"
                f" ({100. * i / len(self.train_dataloader):.0f}%)] "
                f'Loss: {out_criterion["loss"].item():.4f} | '
                f"dist loss: {out_criterion[dloss_name].item():.4f} | "
                f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                f"Aux loss: {aux_loss.item():.2f}"
            )
        return current_step

    def getAvgMeter(self):
        return {
            "loss": AverageMeter(),
            "bpp_loss": AverageMeter(),
            "aux_loss": AverageMeter(),
            "distortion_loss": AverageMeter(),
            "psnr": AverageMeter(),
            "ms_ssim": AverageMeter(),
        }

    def updateAvgMeter(self, avgMeter, out_criterion, rec, gt):
        dloss_name = "mse_loss" if out_criterion["mse_loss"] is not None else "ms_ssim_loss"
        avgMeter["aux_loss"].update(self.net.aux_loss())
        avgMeter["bpp_loss"].update(out_criterion["bpp_loss"])
        avgMeter["loss"].update(out_criterion["loss"])
        avgMeter["distortion_loss"].update(out_criterion[dloss_name])

        p, m = compute_metrics(rec, gt)
        avgMeter["psnr"].update(p)
        avgMeter["ms_ssim"].update(m)

    def validate_log(self, avgMeter, epoch):
        if self.rank == 0:
            self.logger_val.info(
                f"Test epoch {epoch}: Average losses: "
                f"Loss: {avgMeter['loss'].avg:.4f} | "
                f"dist loss: {avgMeter['distortion_loss'].avg:.4f} | "
                f"Bpp loss: {avgMeter['bpp_loss'].avg:.2f} | "
                f"Aux loss: {avgMeter['aux_loss'].avg:.2f} | "
                f"PSNR: {avgMeter['psnr'].avg:.6f} | "
                f"MS-SSIM: {avgMeter['ms_ssim'].avg:.6f}"
            )
            self.tb_logger.add_scalar("{}".format("[val]: loss"), avgMeter["loss"].avg, epoch + 1)
            self.tb_logger.add_scalar("{}".format("[val]: bpp_loss"), avgMeter["bpp_loss"].avg, epoch + 1)
            self.tb_logger.add_scalar("{}".format("[val]: psnr"), avgMeter["psnr"].avg, epoch + 1)
            self.tb_logger.add_scalar("{}".format("[val]: ms-ssim"), avgMeter["ms_ssim"].avg, epoch + 1)
            self.tb_logger.add_scalar(f"[val]: dloss", avgMeter["distortion_loss"].avg, epoch + 1)

    @torch.no_grad()
    def validate_one_epoch(self, epoch):
        self.net.eval()
        save_dir = os.path.join(self.exp_dir_path, "val_images", "%03d" % (epoch + 1))
        os.makedirs(save_dir, exist_ok=True)

        avgMeter = self.getAvgMeter()
        for i, d in enumerate(self.val_dataloader):
            out_criterion, out_net = self.forward(d)
            if isinstance(d, list):
                d = d[1] if self.channel == 1 else d[0]
            self.updateAvgMeter(avgMeter, out_criterion, out_net["x_hat"], d)
            if (i % 20 == 1 or self.debug) and self.rank == 0:
                saveImg(out_net["x_hat"][0], os.path.join(save_dir, "%03d_rec.png" % i))
                saveImg(d[0], os.path.join(save_dir, "%03d_gt.png" % i))
        self.validate_log(avgMeter, epoch)
        return avgMeter["loss"].avg


if __name__ == "__main__":
    pass
