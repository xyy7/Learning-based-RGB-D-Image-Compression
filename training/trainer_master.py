import os
import sys

sys.path.append("..")  # cd xx/playground

import os

import torch

from .trainer import modelZoo
from .trainer_single import TrainerSingle

class TrainerMaster(TrainerSingle):
    def __init__(self, args, model_config) -> None:
        super().__init__(args, model_config)
        self.aux_channel = 3 if args.channel == 1 else 1
        self.aux_net = modelZoo["ELIC"](config=model_config, channel=self.aux_channel, return_mid=True).eval()
        self.ckpt_path1 = args.checkpoint1

        self.train_dataloader, self.val_dataloader = self.init_dataset(
            args.dataset, args.val_dataset, args.batch_size, args.test_batch_size, args.num_workers, 4
        )

    def forward(self, d):
        if self.channel == 1:
            aux = d[0].to(self.device)
            d = d[1].to(self.device)
        else:
            aux = d[1].to(self.device)
            d = d[0].to(self.device)
        # self.logger_train.debug(f'{aux.shape}, {d.shape}')
        with torch.no_grad():
            out = self.aux_net(aux)
        out_net = self.net(d, out["x_hat"], out)
        out_criterion = self.criterion(out_net, d)
        return out_criterion, out_net

    def restore(self, ckpt_path=None, restore_epoch=0):
        epoch = super().restore(ckpt_path, restore_epoch)
        aux_checkpoint = torch.load(self.ckpt_path1)
        self.aux_net.load_state_dict(aux_checkpoint["state_dict"])
        self.aux_net.update(force=True)
        self.aux_net = self.aux_net.to(self.device)
        return epoch


if __name__ == "__main__":
    pass
