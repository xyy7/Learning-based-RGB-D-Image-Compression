from torch.nn import functional as F
from torchvision import transforms


def window_partition(x, window_size=4):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    x = x.permute(0, 2, 3, 1)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    windows = windows.permute(0, 3, 1, 2)

    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    windows = windows.permute(0, 2, 3, 1)
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    x = x.permute(0, 3, 1, 2)
    return x


def pad1(x, p=2**6, mode="reflect"):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        # mode="constant",
        mode=mode,
        value=0,
    )


def pad0(x, p=2**6, mode="reflect"):
    pad_h = 0
    pad_w = 0
    H, W = x.size(2), x.size(3)

    if H % p != 0:
        pad_h = p * (H // p + 1) - H
    if W % p != 0:
        pad_w = p * (W // p + 1) - W
    return F.pad(x, (0, pad_w, 0, pad_h), mode=mode, value=0)


def crop1(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(x, (-padding_left, -padding_right, -padding_top, -padding_bottom), mode="constant", value=0)


def crop0(x, size):
    return x[:, :, 0 : size[0], 0 : size[1]]


def crop(img, padding_mode, size):
    if padding_mode.find("0") != -1:
        img_pad = crop0(img, size)
    else:
        img_pad = crop1(img, size)
    return img_pad


def pad(img, padding_mode, p=2**6):
    if padding_mode.find("CenterCrop") != -1:
        img_pad = transforms.CenterCrop((448, 576))(img)
    else:
        if padding_mode.find("0") != -1:
            img_pad = pad0(img, mode=padding_mode[:-1], p=p)
        else:
            img_pad = pad1(img, mode=padding_mode[:-1], p=p)
    return img_pad
