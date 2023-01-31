import numpy as np
import torch
from utils.config_model import load_checkpoint
import models
from args import parse_args
from utils.misc import get_device, load_config
import data

if __name__ == '__main__':

    args = parse_args()
    load_config(args)
    device = get_device(args)

    model_type = "resnet18"
    model = models.__dict__[args.arch](args)
    model.eval()
    load_checkpoint(model, args)  # need to specify checkpoint in args.checkpoint
    dataset = data.__dict__[args.dataset](args)
    train_loader, val_loader = dataset.data_loaders()

    all_output = None
    for i, data in enumerate(val_loader):
        data_samples, target = data[0].to(device), data[1].to(device)
        # compute output
        output = model(data_samples)
        # merge output in a single data matrix
        if not all_output:
            all_output = output
        else:
            torch.cat((all_output, output), 0)


























