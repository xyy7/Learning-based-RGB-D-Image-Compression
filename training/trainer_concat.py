import os
import sys

sys.path.append("..")  # cd xx/playground

import os

import torch
from utils.IOutils import saveImg
from utils.metrics import AverageMeter, compute_metrics
from utils.rd_loss import RateDistortionLossSingleModal

from .trainer_single import TrainerSingle


class TrainerConcat(TrainerSingle):
    def forward(self, d):
        d = torch.concat([d[0], d[1]], dim=1)
        d = d.to(self.device)
        out_net = self.net(d)
        out_criterion = self.criterion(out_net, d)
        return out_criterion, out_net

    @torch.no_grad()
    def validate_one_epoch(self, epoch):
        self.net.eval()
        save_dir = os.path.join(self.exp_dir_path, "val_images", "%03d" % (epoch + 1))
        os.makedirs(save_dir, exist_ok=True)

        avgMeter = self.getAvgMeter()
        for i, d in enumerate(self.val_dataloader):
            out_criterion, out_net = self.forward(d)
            self.updateAvgMeter(avgMeter, out_criterion, out_net["x_hat"], torch.concat(d, dim=1))
            if (i % 20 == 1 or self.debug) and self.rank == 0:
                saveImg(out_net["x_hat"][0, :3], os.path.join(save_dir, "%03d_rgb_rec.png" % i))
                saveImg(out_net["x_hat"][0, 3:], os.path.join(save_dir, "%03d_depth_rec.png" % i))
                saveImg(d[0][0], os.path.join(save_dir, "%03d_rgb_gt.png" % i))
                saveImg(d[1][0], os.path.join(save_dir, "%03d_depth_gt.png" % i))  # 默认保存成8bit
        self.validate_log(avgMeter, epoch)
        return avgMeter["loss"].avg


if __name__ == "__main__":
    pass
