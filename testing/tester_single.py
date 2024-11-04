import os
import time

import cv2
import torch
from dataset.utils import *
from torchvision import transforms
from utils.IOutils import *
from utils.metrics import AverageMeter, compute_metrics

from .tester import Tester


class TesterSingle(Tester):
    def __init__(self, args, model_config) -> None:
        super().__init__(args, model_config)

    def getAvgMeter(self):
        return {
            "avg_psnr": AverageMeter(),
            "avg_ms_ssim": AverageMeter(),
            "avg_bpp": AverageMeter(),
            "avg_deocde_time": AverageMeter(),
            "avg_encode_time": AverageMeter(),
        }

    def updateAvgMeter(self, avgMeter, p, m, bpp, dec_time, enc_time):
        avgMeter["avg_psnr"].update(p)
        avgMeter["avg_ms_ssim"].update(m)
        avgMeter["avg_bpp"].update(bpp)
        avgMeter["avg_deocde_time"].update(dec_time)
        avgMeter["avg_encode_time"].update(enc_time)

    def get_rec_dir(self, padding=True, padding_mode="reflect0"):
        if not padding:
            rec_dir = os.path.join(self.save_dir, f"{self.epoch}-CenterCrop")
        else:
            rec_dir = os.path.join(self.save_dir, f"{self.epoch}-padding-{padding_mode}")
        depth_rec_path = os.path.join(rec_dir, "depth_rec")
        rgb_rec_path = os.path.join(rec_dir, "rgb_rec")
        self.init_dir([rec_dir, depth_rec_path, rgb_rec_path])
        return rec_dir

    @torch.no_grad()
    def test_model(self, padding_mode="reflect0", padding=True):
        self.net.eval()
        avgMeter = self.getAvgMeter()
        rec_dir = self.get_rec_dir(padding=padding, padding_mode=padding_mode)

        for i, (img, img_name) in enumerate(self.test_dataloader):
            B, C, H, W = img.shape
            img = img.to(self.device)
            img_pad = pad(img, padding_mode, 2**6)
            if C == 1:
                stream_path = os.path.join(rec_dir, "depth_bin")
            else:
                stream_path = os.path.join(rec_dir, "rgb_bin")

            bpp, enc_time = self.compress_one_image(
                net=self.net, x=img_pad, stream_path=stream_path, H=H, W=W, img_name=img_name[0]
            )
            x_hat, dec_time = self.decompress_one_image(
                net=self.net, stream_path=stream_path, img_name=img_name[0], mode=padding_mode
            )
            self.test_save_and_log_perimg(i, x_hat, img, rec_dir, img_name, avgMeter, bpp, dec_time, enc_time)
        self.test_finish_log(avgMeter, rec_dir)

    def test_save_and_log_perimg(self, i, x_hat, img, rec_dir, img_name, avgMeter, bpp, dec_time, enc_time):
        C = x_hat.shape[1]
        p, m = compute_metrics(x_hat, img)
        bpp_psnr = f"{bpp:.4f}_{p:.4f}_"
        if C == 1:
            if rec_dir.find("sun") != -1:
                depth = x_hat * 100000
            else:
                depth = x_hat * 10000
            depth = depth.cpu().squeeze().numpy().astype("uint16")
            cv2.imwrite(os.path.join(rec_dir, "depth_rec", f"{img_name[0]}_{bpp_psnr}_rec_16bit.png"), depth)
        rec = torch2img(x_hat)
        img = torch2img(img)
        if C == 1:
            rec.save(os.path.join(rec_dir, "depth_rec", f"{img_name[0]}_{bpp_psnr}_rec_8bit.png"))
        else:
            rec.save(os.path.join(rec_dir, "rgb_rec", f"{img_name[0]}_{bpp_psnr}_rec.png"))

        self.updateAvgMeter(avgMeter, p, m, bpp, dec_time, enc_time)
        self.logger_test.info(
            f"Image[{i}:{img_name[0]}] | "
            f"Bpp loss: {bpp:.4f} | "
            f"PSNR: {p:.4f} | "
            f"MS-SSIM: {m:.4f} | "
            f"Encoding Latency: {enc_time:.4f} | "
            f"Decoding latency: {dec_time:.4f}"
        )

    def test_finish_log(self, avgMeter, rec_dir):
        self.logger_test.info(
            f"Epoch:[{self.epoch}] | "
            f"Avg Bpp: {avgMeter['avg_bpp'].avg:.7f} | "
            f"Avg PSNR: {avgMeter['avg_psnr'].avg:.7f} | "
            f"Avg MS-SSIM: {avgMeter['avg_ms_ssim'].avg:.7f} | "
            f"Avg Encoding Latency: {avgMeter['avg_encode_time'].avg:.6f} | "
            f"Avg Decoding latency: {avgMeter['avg_deocde_time'].avg:.6f}"
        )
        self.write_test_img_name(os.path.join(rec_dir, "depth_rec"), os.path.join(rec_dir, "test_depth.txt"))
        self.write_test_img_name(os.path.join(rec_dir, "rgb_rec"), os.path.join(rec_dir, "test_rgb.txt"))

    def write_test_img_name(self, dir, file):
        files = os.listdir(dir)
        files.sort()
        with open(file, "w") as f:
            for fn in files:
                f.write(f"{dir}/{fn}\n")

    def compress_one_image(self, net, x, stream_path, H, W, img_name, pad_H=None, pad_W=None):
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            out = net.compress(x)
        torch.cuda.synchronize()
        end_time = time.time()
        shape = out["shape"]
        os.makedirs(stream_path, exist_ok=True)
        output = os.path.join(stream_path, img_name)
        if pad_H is None:
            pad_H = H
            pad_W = W
        with Path(output).open("wb") as f:
            write_uints(f, (pad_H, pad_W))
            write_body(f, shape, out["strings"])

        size = filesize(output)
        bpp = float(size) * 8 / (H * W)
        enc_time = end_time - start_time
        return bpp, enc_time

    def decompress_one_image(self, net, stream_path, img_name, mode="reflect0", return_mid=False):
        output = os.path.join(stream_path, img_name)
        with Path(output).open("rb") as f:
            original_size = read_uints(f, 2)
            strings, shape = read_body(f)
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            out = net.decompress(strings, shape)
        torch.cuda.synchronize()
        end_time = time.time()
        dec_time = end_time - start_time
        x_hat = out["x_hat"]
        if return_mid:
            return x_hat, dec_time, out  # out["x_hat"],out["up1"],out["up2"],out["up3"]
        if mode.find("0") != -1:
            x_hat = crop0(x_hat, original_size)
        else:
            x_hat = crop1(x_hat, original_size)
        return x_hat, dec_time
