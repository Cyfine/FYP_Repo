import os
import shutil

import torch
from sklearn.metrics import precision_score, recall_score, f1_score


# All the things we need to log our training process
def save_checkpoint(state, is_best, result_dir, filename="checkpoint.pth.tar", ):
    torch.save(state, os.path.join(result_dir, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(result_dir, filename),
            os.path.join(result_dir, "model_best.pth.tar"),
        )


def accuracy(output, target, top_k=(1,)):
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []

        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul_(100.0 / batch_size)).cpu().float().item())
        return res


def precision(output, target):
    _, pred = output.max(1)
    return precision_score(target.cpu(), pred.cpu(), average="macro", zero_division=0) * 100


def recall(output, target):
    _, pred = output.max(1)
    return recall_score(target.cpu(), pred.cpu(), average="macro", zero_division=0) * 100


def f1(output, target):
    _, pred = output.max(1)
    return f1_score(target.cpu(), pred.cpu(), average="macro", zero_division=0) * 100


class ProgressLogger:
    def __init__(self, num_batches, prefix=""):
        self.batch_fmtstr = _format_str(num_batches)
        self.trackers = {}
        self.prefix = prefix

    def log(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(tracker[1]) for tracker in self.trackers.items()]
        print("\t".join(entries))
        pass

    def write_tensorboard(self, writer, prefix, global_step):
        for _, trackers in self.trackers.items():
            writer.add_scalar(f"{prefix}/{trackers.name}", trackers.val, global_step)

    def add(self, metric_name: str, metric_format: str = ":f"):
        self.trackers[metric_name] = ScalarTracker(metric_name, metric_format)

    def update(self, metric_name: str, value: float, n: int = 1):
        self.trackers[metric_name].update(value, n)

    def get(self, metric_name: str):
        return self.trackers[metric_name]


def _format_str(num_batches):  # format string for logger
    fmt = "{:" + str(len(str(num_batches // 1))) + "d}"
    return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class ScalarTracker(object):
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.format_str = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def __str__(self):
        fmtstr = "{name} {val" + self.format_str + "} ({avg" + self.format_str + "})"
        return fmtstr.format(name=self.name, val=self.val, avg=self.avg)
