import os
import sys

import torch

DIR = os.path.abspath(os.path.dirname(__file__))
print(DIR)
sys.path.append(os.path.abspath(os.path.dirname(DIR)))
import faulthandler

from config.args import test_options
from config.config import MLIC_model_config, model_config
from PIL import Image, ImageFile
from testing.tester_concat import TesterConcat
from testing.tester_master import TesterMaster
from testing.tester_single import TesterSingle
from testing.tester_united import TesterUnited

faulthandler.enable()
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def main(argv):
    torch.backends.cudnn.deterministic = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    args = test_options(argv)
    if args.model.find("MLIC")!=-1:
        config = MLIC_model_config()
    else:
        config = model_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(args)
    if args.channel == 4:
        if args.model.find("cat") != -1:
            print("TesterConcat")
            tester = TesterConcat(args, config)
        else:
            print("TesterUnited")
            tester = TesterUnited(args, config)
    else:
        if args.model.find("master") != -1:
            print("TesterMaster")
            tester = TesterMaster(args, config)
        else:
            print("TesterSingle")
            tester = TesterSingle(args, config)
    tester.test_model(padding_mode="replicate0", padding=True)

if __name__ == "__main__":
    main(sys.argv[1:])
