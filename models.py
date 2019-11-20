import torch.nn as nn
from torch import cat, sigmoid
import torchvision
import os

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


class ConvDown2D(nn.Module):
    def __init__(self, in_channels, out_channels, p=1, s=1, k=3):
        super(ConvDown2D, self).__init__()
        self.conv = Conv2D(in_channels, out_channels, p, s, k)
        self.dwn = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.dwn(self.conv(x))


class ConvUp2D(nn.Module):
    def __init__(self, in_channels, out_channels, p=1, s=1, k=3):
        super(ConvUp2D, self).__init__()
        self.conv = Conv2D(in_channels, out_channels, p, s, k)
        self.up = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(self.conv(x))


class ConvOut(nn.Module):
    def __init__(self, in_channels, out_channels, p=1, s=1, k=3):
        super(ConvOut, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p, stride=s)

    def forward(self, x):
        return self.conv(x)


class FishNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(FishNet, self).__init__()

        self.c = Conv2D(num_channels, 32)
        self.e1 = ConvDown2D(32, 64)
        self.e2 = ConvDown2D(64, 128)
        self.e3 = ConvDown2D(128, 256)
        self.d3 = ConvUp2D(256, 128)
        self.d2 = ConvUp2D(128, 64)
        self.d1 = ConvUp2D(64, 32)
        self.out = ConvOut(32, num_classes)

    def forward(self, x):
        x = self.c(x)

        x = self.e1(x)
        x = self.e2(x)
        x = self.e3(x)
        x = self.d3(x)
        x = self.d2(x)
        x = self.d1(x)
        return self.out(x)


m = FishNet(1, 2)
torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print('Total Params:', torch_total_params)
