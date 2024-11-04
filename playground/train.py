# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

sys.path.append("..")  # ELIC


from config.args import train_options
from config.config import MLIC_model_config, model_config
from training.trainer_concat import TrainerConcat
from training.trainer_master import TrainerMaster
from training.trainer_single import TrainerSingle
from training.trainer_united import TrainerUnited


def gitIt(exp_name):
    git_add = "git add ."
    git_commit = f"git commit -m {exp_name}"
    os.system(git_add)
    os.system(git_commit)
    print(git_add, git_commit)


def main(argv):
    args = train_options(argv)

    if args.model.find("MLIC")!=-1:
        config = MLIC_model_config()
    else:
        config = model_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if args.git:
        gitIt(args.experiment)

    if args.channel == 4:
        if args.model.find("cat") != -1:
            print("TrainerConcat")
            trainer = TrainerConcat(args, config)
        else:
            print("TrainerUnited")
            trainer = TrainerUnited(args, config)
    else:
        if args.model.find("master") != -1:
            print("TrainerMaster")
            trainer = TrainerMaster(args, config)
        else:
            print("TrainerSingle")
            trainer = TrainerSingle(args, config)
    trainer.fit(
        seed=args.seed, auto_restore=args.auto_restore, ckpt_path=args.checkpoint, restore_epoch=args.start_epoch
    )


if __name__ == "__main__":
    main(sys.argv[1:])
