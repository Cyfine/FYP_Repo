import data
import os

import torch
from torch.utils.tensorboard import SummaryWriter

import models
from args import parse_args
from utils.logging import save_checkpoint
from utils.schedules import get_optimizer, get_lr_schedule
from utils.val import val
from utils.config_model import load_checkpoint, config_model

from train.trainer import train
from train.losses.trades import TradesLoss

from utils.misc import (
    create_checkpoint_dir,
    set_seed,
    get_device,
    copy_directory,
    load_config,
    print_args,
    prepare_data_parallel
)


def main():
    # load configuration file specified by --config
    args = parse_args()
    load_config(args)
    print_args(args)

    # create checkpoint dir
    result_main_dir, result_sub_dir = create_checkpoint_dir(args)

    # set seed
    set_seed(args)

    # get device for training
    device = get_device(args)

    # tensorboard writer
    writer = SummaryWriter(os.path.join(result_sub_dir, "tensorboard"))

    # obtain model
    model = models.__dict__[args.arch](args)

    # configure model
    model_config = config_model(model, args)

    if args.checkpoint: load_checkpoint(model, args)

    model_config.show_require_grad()

    # use data parallel to parallel training
    model = prepare_data_parallel(args, model)

    model.to(device)

    # the data loading logic should be implemented in ./data and returns train_loader, val_loader and test_loader
    dataset = data.__dict__[args.dataset](args)
    train_loader, val_loader = dataset.data_loaders()

    # optimizer is by args.optimizer (--optimizer), support sgd, adam, rmsprop
    optimizer = get_optimizer(model, args)

    # Config Loss function and optimizers
    assert args.loss in ["trades", "cross_entropy"]
    if args.loss == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == "trades":  # use trades loss for adversarial training
        criterion = TradesLoss(model, device, optimizer, args.step_size, args.epsilon, args.steps, args.beta,
                               args.clip_min, args.clip_max, args.distance)

    # adjust the learning rate
    schedule_lr = get_lr_schedule(args.lr_schedule)(optimizer, args)

    # Model training iterations start here
    best_val_acc = 0

    if args.checkpoint:
        val_acc, val_acc5 = val(model, device, val_loader, criterion, args, writer, epoch=0)
        print(f"Loaded Model =>  val_acc: {val_acc}, val_acc5: {val_acc5}")

    for epoch in range(args.epochs + args.warmup_epochs):
        schedule_lr(epoch)  # adjust learning rate

        train(model, device, train_loader, criterion, optimizer, epoch, args, writer)

        # do model validation after each training epoch
        val_acc, val_acc5 = val(model, device, val_loader, criterion, args, writer, epoch=epoch)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        training_state = {"epoch": epoch + 1,
                          "arch": args.arch,
                          "state_dict": model.state_dict(),
                          "best_prec1": val_acc > best_val_acc,
                          "optimizer": optimizer.state_dict(), }

        save_checkpoint(training_state, is_best, os.path.join(result_sub_dir, "checkpoint"))

        print(f"Epoch {epoch}, val_acc: {val_acc}, val_acc5: {val_acc5} best_val_acc: {best_val_acc}")

        copy_directory(result_sub_dir, os.path.join(result_main_dir, "latest_exp"))


if __name__ == "__main__":
    main()
