import torch
from utils.logging import ProgressLogger

import time
from utils.logging import accuracy, precision, recall, f1

import torchvision


def val(model, device, val_loader, criterion, args, writer, epoch=0):
    logger = ProgressLogger(len(val_loader), prefix="Val: ")
    logger.add("Time", ":6.3f")
    logger.add("Loss", ":.4f")
    logger.add("Acc_1", ":6.2f")
    logger.add("Acc_5", ":6.2f")
    logger.add("Precision", ":.4f")
    logger.add("Recall", ":.4f")
    logger.add("F1", ":.4f")

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            data_samples, target = data[0].to(device), data[1].to(device)

            # compute output
            output = model(data_samples)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, (1, 5))
            precision_score = precision(output, target)
            recall_score = recall(output, target)
            f1_score = f1(output, target)

            # update logger
            logger.update("Loss", loss.item(), data_samples.size(0))
            logger.update("Acc_1", acc1, data_samples.size(0))
            logger.update("Acc_5", acc5, data_samples.size(0))
            logger.update("Precision", precision_score, data_samples.size(0))
            logger.update("Recall", recall_score, data_samples.size(0))
            logger.update("F1", f1_score, data_samples.size(0))
            logger.update("Time", time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                logger.log(i)

            if writer:
                # the line plot in tensorboard
                if epoch == "test":
                    logger.write_tensorboard(writer, "test", i)
                else:
                    logger.write_tensorboard(writer, "val", epoch * len(val_loader) + i)

                #  show the image feed to the model, only the first epoch
                if i == 0:
                    writer.add_image("Validation Image",
                                     torchvision.utils.make_grid(data_samples[0: len(data_samples) // 4]), )

    return logger.get("Acc_1").avg, logger.get("Acc_5").avg
