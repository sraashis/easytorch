import torch
import torch.nn.functional as F
from torch import nn


def initialize_model_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class FullConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, p=1, s=1):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p, stride=s),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, include_x=True):
        super().__init__()
        self.include_x = include_x
        self.conv_dwn = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = FullConv(in_channels, out_channels)
        self.conv_up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        _ch = in_channels + out_channels if self.include_x else out_channels
        self.conv_out = FullConv(_ch, out_channels)

    def forward(self, x):
        conv_dwn = self.conv_dwn(x)
        conv = self.conv(conv_dwn)
        conv_up = self.conv_up(conv)
        _ip = torch.cat([x, conv_up], 1) if self.include_x else conv_up
        return self.conv_out(_ip), conv


class SkullNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(SkullNet, self).__init__()
        self.r = 4
        self.num_classes = num_classes
        self.c = FullConv(num_channels, int(16 * self.r))
        self.c1 = BottleNeck(int(16 * self.r), int(16 * self.r))
        self.c2 = BottleNeck(int(16 * self.r), int(16 * self.r))
        self.c3 = BottleNeck(int(16 * self.r), int(16 * self.r))
        self.c4 = BottleNeck(int(16 * self.r), int(16 * self.r))
        self.c5 = BottleNeck(int(16 * self.r), int(16 * self.r))
        self.c6 = BottleNeck(int(16 * self.r), int(16 * self.r))
        self.c7 = BottleNeck(int(16 * self.r), int(16 * self.r))

        self.c1_dwns = FullConv(int(16 * self.r), int(16 * self.r), p=0)
        self.c2_dwns = FullConv(int(16 * self.r), int(15 * self.r), p=0)
        self.c3_dwns = FullConv(int(15 * self.r), int(14 * self.r), p=0, s=2)
        self.c4_dwns = FullConv(int(14 * self.r), int(13 * self.r), p=0)
        self.c5_dwns = FullConv(int(13 * self.r), int(12 * self.r), p=0)
        self.c6_dwns = FullConv(int(12 * self.r), int(11 * self.r), p=0)
        self.c7_dwns = FullConv(int(11 * self.r), int(10 * self.r), p=0, s=2)
        self.c8_dwns = FullConv(int(10 * self.r), int(9 * self.r), p=0)
        self.c9_dwns = FullConv(int(9 * self.r), int(8 * self.r), p=0)
        self.c10_dwns = FullConv(int(8 * self.r), int(7 * self.r), p=0)
        self.c11_dwns = FullConv(int(7 * self.r), int(6 * self.r), p=0)
        self.c12_dwns = FullConv(int(6 * self.r), int(5 * self.r), p=0)
        self.c13_dwns = FullConv(int(5 * self.r), int(4 * self.r), p=0)
        self.c14_dwns = FullConv(int(4 * self.r), int(3 * self.r), p=0)
        self.c15_dwns = FullConv(int(3 * self.r), int(2 * self.r), p=0)
        self.c16_dwns = FullConv(int(2 * self.r), int(1 * self.r), p=0)

        self.out_flat_shape = self.r * 12 * 12
        self.fc1 = nn.Linear(self.out_flat_shape, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc_out = nn.Linear(512, 12)
        initialize_model_weights(self)

    def forward(self, x):
        c = self.c(x)
        c1_b, c1_dwn = self.c1(c)

        c2_b, c2_dwn = self.c2(c1_b)

        c3_b, c3_dwn = self.c3(c2_b)

        c4_b, c4_dwn = self.c4(c3_b)

        c5_b, c5_dwn = self.c5(c4_b)

        c6_b, c6_dwn = self.c6(c5_b)

        c7_b, c7_dwn = self.c7(c6_b)

        c_dwns = self.c1_dwns(c7_dwn)
        c_dwns = self.c2_dwns(c_dwns)
        c_dwns = self.c3_dwns(c_dwns)
        c_dwns = self.c4_dwns(c_dwns)
        c_dwns = self.c5_dwns(c_dwns)
        c_dwns = self.c6_dwns(c_dwns)
        c_dwns = self.c7_dwns(c_dwns)
        c_dwns = self.c8_dwns(c_dwns)
        c_dwns = self.c9_dwns(c_dwns)
        c_dwns = self.c10_dwns(c_dwns)
        c_dwns = self.c11_dwns(c_dwns)
        c_dwns = self.c12_dwns(c_dwns)
        c_dwns = self.c13_dwns(c_dwns)
        c_dwns = self.c14_dwns(c_dwns)
        c_dwns = self.c15_dwns(c_dwns)
        c_dwns = self.c16_dwns(c_dwns)

        print(c_dwns.shape)
        fc1 = self.fc1(c_dwns.view(-1, self.out_flat_shape))
        fc1 = F.relu(self.fc1_bn(fc1), inplace=True)
        fc2 = F.relu(self.fc2_bn(self.fc2(fc1)), inplace=True)
        fc_out = self.fc_out(fc2)
        out = fc_out.view(fc_out.shape[0], 2, -1, 1)
        return out


m = SkullNet(1, 2)
torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print('Total Params:', torch_total_params)
