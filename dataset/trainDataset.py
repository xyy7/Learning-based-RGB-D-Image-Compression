import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BaseDataset(Dataset):
    def __init__(self, train_dir, is_train, channel=4, debug=False):
        self.image_size = 256
        self.train = is_train
        self.channel = channel
        if channel > 1:
            rgb_dir = train_dir + "/color/*"
            print("rgb_dir", rgb_dir)
            self.rgb_files = sorted(glob.glob(rgb_dir))
            if debug:
                self.rgb_files = self.rgb_files[:100]

            self.len = len(self.rgb_files)

        if channel == 1 or channel == 4:
            depth_dir = train_dir + "/gt/*"
            print("depth_dir", depth_dir)
            self.depth_files = sorted(glob.glob(depth_dir))
            # self.depth_max = 255
            if debug:
                self.depth_files = self.depth_files[:100]
            self.len = len(self.depth_files)

    def __getitemForChannel4__(self, index):
        rgb_path = self.rgb_files[index]
        img = Image.open(rgb_path).convert("RGB")
        img = np.array(img) / 255
        img = img.transpose(2, 0, 1)

        depth_path = self.depth_files[index]
        depth = Image.open(depth_path)
        depth_max = 255.0 if np.array(depth).max() < 255 else self.depth_max
        depth = np.array(depth) / depth_max
        if len(depth.shape) == 3:
            depth = depth[0]

        rgb = torch.from_numpy(img)  # [3,H,W]
        depth = torch.from_numpy(depth)  # [H,W]
        depth = torch.unsqueeze(depth, 0)

        # Random crop
        # top: int, left: int, height: int, width: int
        if self.train:
            i, j, h, w = transforms.RandomCrop.get_params(rgb, output_size=(256, 256))
            rgb = TF.crop(rgb, i, j, h, w)
            depth = TF.crop(depth, i, j, h, w)
            # Random horizontal flipping
            if random.random() > 0.5:
                rgb = TF.hflip(rgb)
                depth = TF.hflip(depth)
            # Random vertical flipping
            if random.random() > 0.5:
                rgb = TF.vflip(rgb)
                depth = TF.vflip(depth)
        else:
            transform = transforms.CenterCrop((448, 576))
            rgb = transform(rgb)
            depth = transform(depth)
        rgb = rgb.type(torch.FloatTensor)
        depth = depth.type(torch.FloatTensor)
        return rgb, depth

    def __getitemForChannel3__(self, index):
        rgb_path = self.rgb_files[index]
        img = Image.open(rgb_path).convert("RGB")
        img = np.array(img) / 255
        img = img.transpose(2, 0, 1)
        rgb = torch.from_numpy(img)  # [3,H,W]

        # Random crop
        # top: int, left: int, height: int, width: int
        if self.train:
            i, j, h, w = transforms.RandomCrop.get_params(rgb, output_size=(256, 256))
            rgb = TF.crop(rgb, i, j, h, w)
            # Random horizontal flipping
            if random.random() > 0.5:
                rgb = TF.hflip(rgb)
            # Random vertical flipping
            if random.random() > 0.5:
                rgb = TF.vflip(rgb)
        else:
            transform = transforms.CenterCrop((448, 576))
            rgb = transform(rgb)
        rgb = rgb.type(torch.FloatTensor)
        return rgb

    def __getitemForChannel1__(self, index):
        depth_path = self.depth_files[index]
        depth = Image.open(depth_path)
        depth_max = 255 if np.array(depth).max() < 255 else self.depth_max
        depth = np.array(depth) / depth_max
        if len(depth.shape) == 3:
            depth = depth[0]

        depth = torch.from_numpy(depth)  # [H,W]
        depth = torch.unsqueeze(depth, 0)

        # Random crop
        # top: int, left: int, height: int, width: int
        if self.train:
            i, j, h, w = transforms.RandomCrop.get_params(depth, output_size=(256, 256))
            depth = TF.crop(depth, i, j, h, w)
            # Random horizontal flipping
            if random.random() > 0.5:
                depth = TF.hflip(depth)
            # Random vertical flipping
            if random.random() > 0.5:
                depth = TF.vflip(depth)
        else:
            transform = transforms.CenterCrop((448, 576))
            depth = transform(depth)
        depth = depth.type(torch.FloatTensor)
        return depth

    def __getitem__(self, index):
        if self.channel == 4:
            return self.__getitemForChannel4__(index)
        if self.channel == 3:
            return self.__getitemForChannel3__(index)
        if self.channel == 1:
            return self.__getitemForChannel1__(index)

    def __len__(self):
        return self.len


class nyuv2(BaseDataset):
    def __init__(self, train_dir, is_train, channel=4, debug=False):
        super().__init__(train_dir, is_train, channel, debug)
        self.depth_max = 10000.0


class sun(BaseDataset):
    def __init__(self, train_dir, is_train, channel=4, debug=False):
        super().__init__(train_dir, is_train, channel, debug)
        self.depth_max = 100000.0
