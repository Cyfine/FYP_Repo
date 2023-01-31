from models.resnet import *
import torch


class Net(nn.Module):
    def __init__(self, ll, num_classes, googlenet):
        super().__init__()
        self.googlenet = googlenet
        self.fc1 = ll(1000, 200)
        self.fc2 = ll(200, 84)
        self.fc3 = ll(84, num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.googlenet(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def fc3googlenet(args):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=args.pretrained)
    cl,ll = get_layer(args)
    return Net(ll,args.num_classes,model)