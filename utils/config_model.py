# TODO: configure model for transfer, prune etc
import torch
import math
import os
from torch import nn
from models.layers import SubnetConv, SubnetLinear


def load_checkpoint(model, args):
    if os.path.isfile(args.checkpoint):
        print("Loading source model from '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print("Loaded checkpoint '{}'".format(args.checkpoint))
    else:
        print("No checkpoint found at '{}'".format(args.resume))


def config_model(model, args):
    """
    Configure the model before training
    :param model: model
    :param args: arguments from command line
    :return: SubnetModelConfiger,  the object that contains the model and
    configuration logic if needed in the future
    """
    change_model_output_dim(model, args)
    model_config = SubnetModelConfiger(model, args)
    if args.exp_mode == "train":
        model_config.config_train_mode()
    elif args.exp_mode == "prune":
        model_config.config_prune_mode()
    elif args.exp_mode == "finetune" :
        model_config.config_finetune_mode()
    return model_config



class SubnetModelConfiger:
    def __init__(self, model, args):
        self.model = model
        self.args = args

    def config_prune_mode(self):
        print(">>>>>>>>> CONFIG PRUNE <<<<<<<<<<")
        self.freeze_vars("weight")
        self.freeze_vars("bias")
        self._initialize_scaled_score()
        self.unfreeze_vars("popup_scores")
        print(f"Set pruning rate{self.args.prune_rate}")
        self.config_prune_rate(self.args.prune_rate)

    def config_train_mode(self):
        print(">>>>>>>>> CONFIG NORMAL TRAIN <<<<<<<<<<")
        self.unfreeze_vars("weight")
        self.unfreeze_vars("bias")
        # freeze edge popup score, this one is for model pruning
        self.freeze_vars("popup_scores")
        
    def config_finetune_mode(self):
        print(">>>>>>>>> CONFIG FINETUNE <<<<<<<<<<")
        self.unfreeze_vars("weight")
        self.unfreeze_vars("bias")
        self.freeze_vars("popup_scores")
        print(f"Use pruning rate{self.args.prune_rate}")
        self.config_prune_rate(self.args.prune_rate)
        
    def config_prune_rate(self, k):
        for _, v in self.model.named_modules():
            if hasattr(v, "set_prune_rate"):
                v.set_prune_rate(k)

    # There are multiple ways to initialize the score of the score for edge pop up
    # here we use the initialization from  https://dl.acm.org/doi/10.5555/3495724.3497373
    def _initialize_scaled_score(self):
        print("Initialization relevance score proportional to weight magnitudes (OVERWRITING SOURCE NET SCORES)")
        for m in self.model.modules():
            if hasattr(m, "popup_scores"):
                n = nn.init._calculate_correct_fan(m.popup_scores, "fan_in")
                m.popup_scores.data = (math.sqrt(6 / n) * m.weight.data / torch.max(torch.abs(m.weight.data)))

    def freeze_vars(self, var_name):
        assert var_name in ["weight", "bias", "popup_scores"]
        for i, v in self.model.named_modules():
            if hasattr(v, var_name):
                if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if getattr(v, var_name) is not None:
                        getattr(v, var_name).requires_grad = False

    def unfreeze_vars(self, var_name):
        assert var_name in ["weight", "bias", "popup_scores"]
        for i, v in self.model.named_modules():
            if hasattr(v, var_name):
                if getattr(v, var_name) is not None:
                    getattr(v, var_name).requires_grad = True

    def show_require_grad(self):
        for i, v in self.model.named_parameters():
            print(f"variable = {i}, Gradient requires_grad = {v.requires_grad}")


def change_model_output_dim(model, args):
    params = list(model.named_modules())
    _, linear_layer = get_layer(args)
    new_linear = linear_layer(params[-1][1].in_features, args.num_classes)
    setattr(model, params[-1][0], new_linear)


def get_layer(args):
    if args.layer_type == "dense":
        return nn.Conv2d, nn.Linear
    elif args.layer_type == "subnet":
        return SubnetConv, SubnetLinear
