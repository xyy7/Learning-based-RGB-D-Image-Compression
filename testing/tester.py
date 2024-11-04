import os
import sys

sys.path.append("..")  # cd xx/playground

import logging
import os

import torch
from dataset.testDataset import ImageFolder, ImageFolderUnited
from models import modelZoo
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.logger import setup_logger


class Tester:
    def __init__(self, args, model_config) -> None:
        self.device = "cuda"
        self.channel = args.channel
        self.quality = args.quality
        self.debug = args.debug

        if args.experiment is not None and args.experiment != "":
            self.exp_name = args.experiment
        else:
            self.exp_name = self.get_exp_name(args.dataset, args.channel, args.model, args.quality)

        if self.debug:
            self.exp_dir_path = os.path.join("../experiments_test", self.exp_name)
        else:
            self.exp_dir_path = os.path.join("../experiments", self.exp_name)
        self.ckpt_dir_path = os.path.join(self.exp_dir_path, "checkpoints")

        self.model_name = args.model
        self.model_config = model_config
        self.epoch = self.get_net(model_config=model_config, model_name=args.model, ckpt_path=args.checkpoint)

        self.logger_test = self.init_logger(self.exp_dir_path, self.exp_name, self.epoch)
        self.log_init_info(args, model_config)
        self.test_dataloader = self.init_dataset(args.dataset, args.test_batch_size, args.num_workers, self.channel)

        if self.debug:
            self.save_dir = os.path.join("../experiments_test", self.exp_name, "codestream")
        else:
            self.save_dir = os.path.join("../experiments", self.exp_name, "codestream")
        os.makedirs(self.save_dir, exist_ok=True)

    def log_init_info(self, args, config):
        total_params = sum(p.numel() for p in self.net.parameters())
        self.logger_test.info(f"name:{self.exp_name}, params:{total_params}")
        self.logger_test.info(f"args: {args}")
        self.logger_test.info(f"config: {config}")

    def get_net(self, model_config, model_name, ckpt_path):
        for name, model in modelZoo.items():
            if model_name.find(name) != -1:
                self.net = model(config=model_config, channel=self.channel).eval()
                break
        print("checking best path is:", os.path.join(self.ckpt_dir_path, "checkpoint_best_loss.pth.tar"), ckpt_path)
        if ckpt_path is None and os.path.exists(os.path.join(self.ckpt_dir_path, "checkpoint_best_loss.pth.tar")):
            ckpt_path = os.path.join(self.ckpt_dir_path, "checkpoint_best_loss.pth.tar")
        start_epoch = self.restore(ckpt_path)
        return start_epoch

    def get_exp_name(self, dataset=None, channel=3, model_name=None, quality=1):
        if channel == 1:
            modal = "depth_"
        elif channel == 3:
            modal = "rgb_"
        elif channel == 4:
            modal = ""
        dataset_name = self.get_dataset_name(dataset)
        exp_name = f"{dataset_name}_{modal}{model_name}_{quality}"
        return exp_name

    def get_dataset_name(self, dataset):
        if dataset.find("nyu") != -1:
            return "nyuv2"
        return "sunrgbd"

    def init_dataset(self, test_dataset, test_batch_size, num_workers, channel):
        test_transforms = transforms.Compose([transforms.ToTensor()])
        test_dataset = ImageFolder(test_dataset, channel=channel, transform=test_transforms, debug=self.debug)
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=num_workers, shuffle=False)
        return test_dataloader

    def init_dir(self, dirs):
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def init_logger(self, exp_dir_path, exp_name, epoch):
        log_level = logging.DEBUG if self.debug else logging.INFO
        print(f"test_epoch{epoch}" + exp_name)
        setup_logger("test", exp_dir_path, f"test_epoch{epoch}_" + exp_name, level=log_level)
        logger_test = logging.getLogger("test")
        logger_test.info(f"Start testing!")
        return logger_test

    def restore(self, ckpt_path=None):
        print("restore from:", ckpt_path)
        self.ckpt_path = ckpt_path
        checkpoint = torch.load(ckpt_path)
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net.update(force=True)
        self.net = self.net.to(self.device)
        print("epoch:", checkpoint["epoch"])
        return checkpoint["epoch"]

    def test_model(self):
        pass


if __name__ == "__main__":
    pass
