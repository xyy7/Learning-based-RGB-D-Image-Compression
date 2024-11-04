import json
import shutil
import struct
from pathlib import Path

import PIL.Image as Image
import torch
from torchvision.transforms import ToPILImage
import os

""" configuration json """


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, "r") as f:
            config = json.loads(f.read())
            return Config(config)


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 1


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
    # print(len(values))
    return len(values) * 1


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def read_body(fd):
    lstrings = []
    shape = read_uints(fd, 2)
    n_strings = read_uints(fd, 1)[0]
    for _ in range(n_strings):
        # s = read_bytes(fd, read_uints(fd, 1)[0])
        # lstrings.append([s]) # 读进来的是个列表

        num = read_uints(fd, 1)[0]
        s = []
        for _ in range(num):
            ss = read_bytes(fd, read_uints(fd, 1)[0])
            s.append(ss)
        lstrings.append(s)

    return lstrings, shape


def write_body(fd, shape, out_strings):
    bytes_cnt = 0
    bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings)))
    for s in out_strings:
        # bytes_cnt += write_uints(fd, (len(s[0]),))  # y的形状，可以根据z来创建，所以可以融合在一起
        # bytes_cnt += write_bytes(fd, s[0])

        bytes_cnt += write_uints(fd, (len(s),))
        for ss in s:  # 能否处理矩阵？
            bytes_cnt += write_uints(fd, (len(ss),))
            bytes_cnt += write_bytes(fd, ss)  # 能够兼容，只有batchsize=1的时候
        # bytes_cnt += write_bytes(fd, s)  # 读出来的时候怎么恢复,不能直接处理
    return bytes_cnt


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1).squeeze())


def saveImg(x, path):
    img = ToPILImage()(x.clamp_(0, 1).squeeze())
    img.save(path)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        best_filename = filename.replace(filename.split("/")[-1], "checkpoint_best_loss.pth.tar")
        shutil.copyfile(filename, best_filename)


def del_checkpoint(filename="checkpoint.pth.tar"):
    if os.path.exists(filename):
        os.remove(filename)
