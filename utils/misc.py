import os
import random
import sys
from distutils.dir_util import copy_tree
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(args):
    arg_override = []

    for arg in sys.argv:
        arg_name = get_arg(arg)
        if arg.startswith("-") and arg_name != "config":
            arg_override.append(arg_name)

    yaml_file = open(args.configs).read()

    # override args
    loaded_yaml = yaml.load(yaml_file, Loader=yaml.FullLoader)
    for v in arg_override:
        loaded_yaml[v] = getattr(args, v)

    print(f"Loading configuration from: {args.configs}")
    args.__dict__.update(loaded_yaml)


def get_arg(arg: str):
    # omit '--' in the argument
    i = 0
    while arg[i] == "-":
        i += 1
    arg = arg[i:]

    arg = arg.replace("-", "_")

    return arg.split("=")[0]


def print_args(args):
    print(">>>>>>>>> Experiment Configurations <<<<<<<<<")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print(">>>>>>>>> Experiment Configurations End <<<<<<<<<\n")

def create_checkpoint_dir(args):
    """
    create folder for storing checkpoints and tensorboard logs
    """
    result_main_dir = os.path.join(Path(args.result_dir), args.exp_name)
    if os.path.exists(result_main_dir):
        n = len(next(os.walk(result_main_dir))[-2])  # prev experiments with same name
        result_sub_dir = os.path.join(
            result_main_dir,
            "{}--lr-{}_epochs-{}_warmuplr-{}_warmupepochs-{}".format(
                n + 1,
                args.lr,
                args.epochs,
                args.warmup_lr,
                args.warmup_epochs,
            ),
        )
    else:
        os.makedirs(result_main_dir, exist_ok=True)
        result_sub_dir = os.path.join(
            result_main_dir,
            "1--lr-{}_epochs-{}_warmuplr-{}_warmupepochs-{}".format(
                args.lr,
                args.epochs,
                args.warmup_lr,
                args.warmup_epochs,
            ),
        )

    os.mkdir(result_sub_dir)
    os.mkdir(os.path.join(result_sub_dir, "checkpoint"))

    return result_main_dir, result_sub_dir


def get_device(args):
    """
    get device for training
    """
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    if use_cuda:
        print(f"Use cuda:{gpu_list[0]}")
    else:
        print("use cpu")
    return torch.device(f"cuda:{gpu_list[0]}" if use_cuda else "cpu")


def prepare_data_parallel(args, model):
    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    if len(gpu_list) > 1:
        print("Use Multiple GPUs")
        print("GPU List: " + str(gpu_list))
        model = torch.nn.DataParallel(model, device_ids=gpu_list)
    return model
    


def set_seed(args):
    """
    set seeds for result reproducibility
    """
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def copy_directory(src, dest):
    if not os.path.exists(dest):
        os.mkdir(dest)
    copy_tree(src, dest)
    





