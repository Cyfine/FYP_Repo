import time

import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from utils.logging import ProgressLogger
from utils.logging import accuracy, precision, recall, f1


def train(model, device, train_loader, criterion, optimizer, epoch, args, writer):
    print(" ->->->->->->->->->-> ONE EPOCH TRAINING <-<-<-<-<-<-<-<-<-<-")

    logger = ProgressLogger(len(train_loader), prefix=f"Epoch: [{epoch}]")
    logger.add("Time", ":6.3f")
    logger.add("Loss", ":.4f")
    logger.add("Acc_1", ":6.2f")
    logger.add("Acc_5", ":6.2f")
    logger.add("Precision", ":.4f")
    logger.add("Recall", ":.4f")
    logger.add("F1", ":.4f")

    model.train()
    end = time.time()

    for i, data in enumerate(train_loader):
        input_data, target = data[0], data[1]
        input_data = input_data.to(device)
        target = target.to(device)

        # basic properties of training
        if i == 0:
            print(
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )

        output = model(input_data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, (1, 5))
        precision_score = precision(output, target)
        recall_score = recall(output, target)
        f1_score = f1(output, target)

        # update logger
        logger.update("Loss", loss.item(), input_data.size(0))
        logger.update("Acc_1", acc1, input_data.size(0))
        logger.update("Acc_5", acc5, input_data.size(0))
        logger.update("Precision", precision_score, input_data.size(0))
        logger.update("Recall", recall_score, input_data.size(0))
        logger.update("F1", f1_score, input_data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        logger.update("Time", time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.log(i)
            logger.write_tensorboard(writer, "train", epoch * len(train_loader) + i)
