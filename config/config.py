import torch.nn as nn
from utils.IOutils import Config


def model_config():
    config = Config(
        {"N": 192, "M": 320, "slice_num": 5, "context_window": 5, "slice_ch": [16, 16, 32, 64, 192], "quant": "ste"}
    )

    return config

def MLIC_model_config():
    config = Config({
        # MLIC and MLIC+
        "N": 192,
        "M": 320,
        "slice_num": 10,
        "context_window": 5,
        "act": nn.GELU,
    })

    return config
