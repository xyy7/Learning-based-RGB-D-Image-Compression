from typing import Tuple

import numpy as np
import torch
from pytorch_msssim import ms_ssim


def compute_metrics(a, b, max_val: float = 1) -> Tuple[float, float]:
    a = a.clamp(0, 1).cuda()
    b = b.clamp(0, 1).cuda()
    mse = torch.mean((a - b) ** 2).item()
    p = 20 * np.log10(max_val) - 10 * np.log10(mse)
    m = ms_ssim(a, b, data_range=max_val).item()
    return p, m


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
