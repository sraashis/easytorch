import torch.nn as nn
from torch import cat


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
        x = self.conv(x)
        return x, self.dwn(x)


class ConvUp2D(nn.Module):
    def __init__(self, in_channels, out_channels, p=1, s=1, k=3):
        super(ConvUp2D, self).__init__()
        self.conv = Conv2D(in_channels, out_channels, p, s, k)
        self.up = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        return x, self.up(x)


class ConvOut(nn.Module):
    def __init__(self, in_channels, out_channels, p=1, s=1, k=3):
        super(ConvOut, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p, stride=s)

    def forward(self, x):
        return self.conv(x)


class FishNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(FishNet, self).__init__()

        self.c1 = Conv2D(num_channels, 64)
        self.c2 = Conv2D(64, 128)
        self.c3 = Conv2D(128, 256)
        self.c4 = Conv2D(256, 128)
        self.c5 = Conv2D(128, 64)
        self.out = ConvOut(64, num_classes)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        return self.out(x)


m = FishNet(1, 2)
torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print('Total Params:', torch_total_params)
