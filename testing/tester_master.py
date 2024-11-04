import os
import time

import torch
from dataset.utils import *
from utils.IOutils import *

from .tester import modelZoo
from .tester_united import TesterUnited


class TesterMaster(TesterUnited):
    def __init__(self, args, model_config) -> None:
        self.ckpt_path1 = args.checkpoint1
        self.aux_channel = 3 if args.channel == 1 else 1
        self.aux_net = modelZoo["ELIC"](config=model_config, channel=self.aux_channel, return_mid=True).eval()
        super().__init__(args, model_config)

    def restore(self, ckpt_path=None):
        epoch = super().restore(ckpt_path)
        aux_checkpoint = torch.load(self.ckpt_path1)
        self.aux_net.load_state_dict(aux_checkpoint["state_dict"])
        self.aux_net.update(force=True)
        self.aux_net = self.aux_net.to(self.device)
        return epoch

    @torch.no_grad()
    def test_model(self, padding_mode="reflect0", padding=True):
        self.net.eval()
        avgMeter = self.getAvgMeter()
        rec_dir = self.get_rec_dir(padding=padding, padding_mode=padding_mode)

        for i, (rgb, depth, rgb_img_name, depth_img_name) in enumerate(self.test_dataloader):
            rgb = rgb.to(self.device)
            depth = depth.to(self.device)
            img_name = rgb_img_name
            if rec_dir.find("depth") != -1:
                img = depth
                aux = rgb
                aux_bin = os.path.join(rec_dir, "rgb_bin")
                master_bin = os.path.join(rec_dir, "depth_bin")
            else:
                img = rgb
                aux = depth
                aux_bin = os.path.join(rec_dir, "depth_bin")
                master_bin = os.path.join(rec_dir, "rgb_bin")

            img_pad = pad(img, padding_mode=padding_mode, p=2**6)
            aux_pad = pad(aux, padding_mode=padding_mode, p=2**6)

            B, C, H, W = img.shape

            aux_bpp, aux_enc_time = self.compress_one_image(
                net=self.aux_net, x=aux_pad, stream_path=aux_bin, H=H, W=W, img_name=img_name[0]
            )
            aux_hat, aux_dec_time, aux_out = self.decompress_one_image(
                net=self.aux_net, stream_path=aux_bin, img_name=img_name[0], mode=padding_mode, return_mid=True
            )

            bpp, enc_time, beta, gamma = self.compress_master(
                net=self.net,
                x=img_pad,
                aux=aux_hat,
                aux_out=aux_out,
                stream_path=master_bin,
                H=H,
                W=W,
                img_name=img_name[0],
            )
            x_hat, dec_time = self.decompress_master(
                net=self.net,
                aux=aux_hat,
                aux_out=aux_out,
                beta=beta,
                gamma=gamma,
                stream_path=master_bin,
                img_name=img_name[0],
                mode=padding_mode,
            )

            aux_hat = crop(aux_hat, padding_mode, (H, W))
            if C == 1:
                rgb_x_hat = aux_hat
                depth_x_hat = x_hat
                rgb_bpp = aux_bpp
                depth_bpp = bpp
            else:
                rgb_x_hat = x_hat
                depth_x_hat = aux_hat
                rgb_bpp = bpp
                depth_bpp = aux_bpp

            self.test_save_and_log_perimg(
                i,
                rgb_x_hat,
                depth_x_hat,
                rgb,
                depth,
                rec_dir,
                rgb_img_name,
                avgMeter,
                rgb_bpp,
                depth_bpp,
                dec_time + aux_dec_time,
                enc_time + aux_enc_time,
            )
        self.test_finish_log(avgMeter, rec_dir)

    def compress_master(self, net, x, aux, aux_out, stream_path, H, W, img_name, padH=None, padW=None):
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            out = net.compress(x, aux, aux_out)
        torch.cuda.synchronize()
        end_time = time.time()
        shape = out["shape"]
        os.makedirs(stream_path, exist_ok=True)
        output = os.path.join(stream_path, img_name)
        if padH is None:
            padH = H
            padW = W
        with Path(output).open("wb") as f:
            write_uints(f, (padH, padW))
            write_body(f, shape, out["strings"])

        size = filesize(output) + 128  # 64个4字节float的beta+64个4字节float的gamma
        bpp = float(size) * 8 / (H * W * len(x) ** 2)
        enc_time = end_time - start_time
        return bpp, enc_time, out["beta"], out["gamma"]

    def decompress_master(self, net, aux, aux_out, beta, gamma, stream_path, img_name, mode="reflect0"):
        output = os.path.join(stream_path, img_name)
        with Path(output).open("rb") as f:
            original_size = read_uints(f, 2)
            strings, shape = read_body(f)
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            out = net.decompress(strings, shape, beta, gamma, aux, aux_out)
        torch.cuda.synchronize()
        end_time = time.time()
        dec_time = end_time - start_time
        x_hat = out["x_hat"]
        if mode.find("0") != -1:
            x_hat = crop0(x_hat, original_size)
        else:
            x_hat = crop1(x_hat, original_size)
        return x_hat, dec_time
