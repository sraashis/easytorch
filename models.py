import torch
import torch.nn.functional as F
from torch import nn

import torch.nn as nn


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, p=1, s=1, k=3):
        super(Conv2D, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p, stride=s),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class Net(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(Net, self).__init__()
        self.c1 = Conv2D(num_channels, num_classes)
        initialize_weights(self)

    def forward(self, x):
        return x
