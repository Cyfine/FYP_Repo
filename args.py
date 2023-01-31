import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training")

    # primary
    parser.add_argument(
        "--configs", type=str, default="", help="configs file",
    )
    parser.add_argument(
        "--result-dir",
        default="./trained_models",
        type=str,
        help="directory to save results",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        help="Name of the experiment (creates dir with this name in --result-dir)",
    )

    parser.add_argument(
        "--exp-mode",
        type=str,
        default="train",
        choices=("train", "prune"),
        help="Train networks following one of these methods.",
    )

    # Model
    parser.add_argument("--arch", type=str, help="Model architecture")

    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of classes",
    )

    # load the check point of model
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to checkpoint",
    )

    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Path to checkpoint",
    )

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset for training and eval"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Root dir for all datasets",
        default="datasets"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)",
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for testing (default: 32)",
    )

    parser.add_argument(
        "--epochs", type=int, default=100, metavar="N", help="number of epochs to train"
    )
    parser.add_argument(
        "--optimizer", type=str, default="sgd", choices=("sgd", "adam", "rmsprop")
    )

    parser.add_argument("--wd", default=1e-4, type=float, help="Weight decay")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")

    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=("step", "cosine"),
        help="Learning rate schedule",
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")

    parser.add_argument(
        "--warmup-epochs", type=int, default=0, help="Number of warmup epochs"
    )
    parser.add_argument(
        "--warmup-lr", type=float, default=0.1, help="warmup learning rate"
    )

    # Additional
    parser.add_argument(
        "--gpu", type=str, default="0", help="Specify which gpu to use"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument("--seed", type=int, default=1234, help="random seed")

    parser.add_argument(
        "--print-freq",
        type=int,
        default=100,
        help="Number of batches to wait before printing training logs",
    )

    # prune rate
    parser.add_argument(
        "--prune-rate",
        type=float,
        default=1,
        help="prune rate k, only k% of the weights are preserved"
    )

    parser.add_argument(
        "--layer-type",
        type=str,
        default="dense",
        choices=("dense", "subnet"),
        help="Layer type to use: dense is normal pytorch layers, "
             "while the subnet is to use custom layer that support pruning logic"
    )

    parser.add_argument(
        "--loss",
        type=str,
        default="cross_entropy",
        help="Loss function to use"
    )
    return parser.parse_args()
