import argparse


def train_options(argv):
    parser = argparse.ArgumentParser(description="Training script.")
    parser.add_argument("-exp", "--experiment", default=None, type=str, required=False, help="Experiment name")
    parser.add_argument("-m", "--model", default="ELIC", help="Model architecture (default: %(default)s)")
    parser.add_argument("-d", "--dataset", default="/data/xyy/nyu5k", type=str, required=False, help="Training dataset")
    parser.add_argument(
        "-vd", "--val_dataset", default="/data/xyy/nyu5k/val", type=str, required=False, help="Training dataset"
    )
    parser.add_argument("-e", "--epochs", default=400, type=int, help="Number of epochs (default: %(default)s)")
    parser.add_argument(
        "-wr", "--warmup_step", default=0, type=int, help="Number of warmup step (default: %(default)s)"
    )
    parser.add_argument("--start_epoch", default=0, type=int, help="Number of restore epochs (default: %(default)s)")
    parser.add_argument("-ch", "--channel", default=3, type=int, help="Number of image channel (default: %(default)s)")
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate (default: %(default)s)")
    parser.add_argument("--lr_scheduler", default="MultiStepLR", type=str, help="lr_scheduler (default: %(default)s)")
    parser.add_argument("-n", "--num-workers", type=int, default=8, help="Dataloaders threads (default: %(default)s)")

    parser.add_argument("--metrics", type=str, default="mse", help="Optimized for (default: %(default)s)")
    parser.add_argument(
        "--distortionLossForDepth", type=str, default="d_loss", help="Optimized for (default: %(default)s)"
    )
    parser.add_argument("-q", "--quality", type=str, default="3_3", help="Quality")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: %(default)s)")
    parser.add_argument("--test-batch-size", type=int, default=1, help="Test batch size (default: %(default)s)")
    parser.add_argument("--aux-learning-rate", default=1e-3, help="Auxiliary loss learning rate (default: %(default)s)")
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID")
    parser.add_argument("--cuda", default=True, help="Use cuda")
    parser.add_argument("--save", action="store_true", help="Save model to disk")
    parser.add_argument("--seed", type=float, default=42, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm", default=1.0, type=float, help="gradient clipping max norm (default: %(default)s"
    )
    parser.add_argument("-c", "--checkpoint", default=None, type=str, help="pretrained model path")
    parser.add_argument("-c1", "--checkpoint1", default=None, type=str, help="pretrained aux model path")
    parser.add_argument("--git", action="store_true", help="Use git to save code")
    parser.add_argument("--auto_restore", action="store_true", help="Restore ckt automatically")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--dist", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args(argv)
    return args


def test_options(argv):
    parser = argparse.ArgumentParser(description="Testing script.")
    parser.add_argument("-exp", "--experiment", default="", type=str, required=False, help="Experiment name")
    parser.add_argument("--channel", default=3, type=int, required=False, help="channel:1/3/4,depth/rgb/both")
    parser.add_argument(
        "-d", "--dataset", default="/data/xyy/nyu5k/nyuv2/test", type=str, required=False, help="Training dataset"
    )
    parser.add_argument("-m", "--model", default="ELIC", help="Model architecture (default: %(default)s)")
    parser.add_argument("-n", "--num-workers", type=int, default=1, help="Dataloaders threads (default: %(default)s)")
    parser.add_argument("--metrics", type=str, default="mse", help="Optimized for (default: %(default)s)")
    parser.add_argument(
        "--test-batch-size", type=int, default=1, help="Test batch size (default: %(default)s)"  # 需要保存图片，计算时间等操作
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("-q", "--quality", type=str, default="1", help="Quality (default: %(default)s)")
    parser.add_argument("-c", "--checkpoint", default=None, type=str, help="pretrained model path")
    parser.add_argument("-c1", "--checkpoint1", default=None, type=str, help="pretrained aux model path")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    return args
