# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet18_Weights, ResNet101_Weights, ResNet152_Weights, ResNet34_Weights 
from models.layers import SubnetConv, SubnetLinear


'''
The implementation of ResNet is the same as the torchvision 
implementation, except that we can use modified conv and subnet layer
to support the pruning logic (adding a mask to the model).
'''


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, conv_layer, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_layer(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_layer(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                conv_layer(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class BasicBlock_dropout_01(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, conv_layer, stride=1):
        super(BasicBlock_dropout_01, self).__init__()
        self.conv1 = conv_layer(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_layer(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                conv_layer(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        self.dropout = nn.Dropout(0.1) # add dropout

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        out = self.dropout(out) # add dropout
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, conv_layer, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv_layer(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_layer(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv_layer(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv_layer(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, conv_layer: nn.Conv2d, linear_layer, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv_layer = conv_layer

        self.conv1 = conv_layer(3, 64, kernel_size=7, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = linear_layer(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.conv_layer, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def resnet18(args, **kwargs):
    conv2d, linear = get_layer(args)
    model = ResNet(conv2d, linear, BasicBlock, [2, 2, 2, 2], **kwargs)

    if args.pretrained:
        pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).state_dict()
        model.load_state_dict(pretrained, strict=False)
    return model

def resnet18_dropout01(args, **kwargs):
    conv2d, linear = get_layer(args)
    model = ResNet(conv2d, linear, BasicBlock_dropout_01, [2, 2, 2, 2], **kwargs)

    if args.pretrained:
        pretrained = models.resnet18(weights="IMAGENET1K_V1").state_dict()
        model.load_state_dict(pretrained, strict=False)
    return model


def resnet34(args, **kwargs):
    conv2d, linear = get_layer(args)
    model = ResNet(conv2d, linear, BasicBlock, [3, 4, 6, 3], **kwargs)

    if args.pretrained:
        pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True).state_dict()
    return model


def resnet50(args, **kwargs):
    conv2d, linear = get_layer(args)
    model = ResNet(conv2d, linear, Bottleneck, [3, 4, 6, 3], **kwargs)

    if args.pretrained:
        pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).state_dict()
        model.load_state_dict(pretrained, strict=False)
    return model


def resnet101(args, **kwargs):
    conv2d, linear = get_layer(args)
    model = ResNet(conv2d, linear, Bottleneck, [3, 4, 23, 3], **kwargs)
    if args.pretrained:
        pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True).state_dict()
        model.load_state_dict(pretrained, strict=False)
    return model


def resnet152(args, **kwargs):
    conv2d, linear = get_layer(args)
    model = ResNet(conv2d, linear, Bottleneck, [3, 8, 36, 3], **kwargs)

    if args.pretrained:
        pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True).state_dict()
        model.load_state_dict(pretrained, strict=False)
    return model


def get_layer(args):
    if args.layer_type == "dense":
        return nn.Conv2d, nn.Linear
    elif args.layer_type == "subnet":
        return SubnetConv, SubnetLinear


if __name__ == '__main__':
    import torchvision
    from layers import SubnetConv, SubnetLinear

    m2 = torchvision.models.resnet18(pretrained=True)
    m1 = resnet18(SubnetConv, SubnetLinear)
    m1.load_state_dict(m2.state_dict(), strict=False)



# def resnet18(args, **kwargs):
#     conv2d, linear = get_layer(args)
#     model = ResNet(conv2d, linear, BasicBlock, [2, 2, 2, 2], **kwargs)

#     if args.pretrained:
#         pretrained = models.resnet18(weights="IMAGENET1K_V1").state_dict()
#         model.load_state_dict(pretrained, strict=False)
#     return model


# def resnet34(args, **kwargs):
#     conv2d, linear = get_layer(args)
#     model = ResNet(conv2d, linear, BasicBlock, [3, 4, 6, 3], **kwargs)

#     if args.pretrained:
#         pretrained = models.resnet34(weights="IMAGENET1K_V1").state_dict()
#         model.load_state_dict(pretrained, strict=False)
#     return model


# def resnet50(args, **kwargs):
#     conv2d, linear = get_layer(args)
#     model = ResNet(conv2d, linear, Bottleneck, [3, 4, 6, 3], **kwargs)

#     if args.pretrained:
#         pretrained = models.resnet50(weights="IMAGENET1K_V2").state_dict()
#         model.load_state_dict(pretrained, strict=False)
#     return model


# def resnet101(args, **kwargs):
#     conv2d, linear = get_layer(args)
#     model = ResNet(conv2d, linear, Bottleneck, [3, 4, 23, 3], **kwargs)
#     if args.pretrained:
#         pretrained = models.resnet101(weights="IMAGENET1K_V2").state_dict()
#         model.load_state_dict(pretrained, strict=False)
#     return model


# def resnet152(args, **kwargs):
#     conv2d, linear = get_layer(args)
#     model = ResNet(conv2d, linear, Bottleneck, [3, 8, 36, 3], **kwargs)

#     if args.pretrained:
#         pretrained = models.resnet152(weights="IMAGENET1K_V2").state_dict()
#         model.load_state_dict(pretrained, strict=False)
#     return model


