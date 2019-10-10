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


class _DoubleConvolution(nn.Module):
    def __init__(self, in_channels, middle_channel, out_channels, k=3, p=0, s=1):
        super(_DoubleConvolution, self).__init__()
        layers = [
            nn.Conv2d(in_channels, middle_channel, kernel_size=k, padding=p, stride=s),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channel, out_channels, kernel_size=k, padding=p, stride=s),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class SkullNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(SkullNet, self).__init__()
        self.r = 2
        self.num_classes = num_classes

        self.C1 = _DoubleConvolution(num_channels, int(64 / self.r), int(64 / self.r))
        self.C2 = _DoubleConvolution(int(64 / self.r), int(128 / self.r), int(128 / self.r), s=2)
        self.C3 = _DoubleConvolution(int(128 / self.r), int(256 / self.r), int(256 / self.r))
        self.C4 = _DoubleConvolution(int(256 / self.r), int(512 / self.r), int(256 / self.r), s=2)
        self.C5 = _DoubleConvolution(int(256 / self.r), int(128 / self.r), int(64 / self.r))

        self.c5_out_dropout = nn.Dropout2d(p=0.5)
        self.c5_out_flat_shape = int(64 / self.r) * 11 * 11
        self.fc1 = nn.Linear(self.c5_out_flat_shape, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_out = nn.Linear(512, 12)
        initialize_model_weights(self)

    def forward(self, x):
        c1 = self.C1(x)

        c2 = self.C2(c1)

        c3 = self.C3(c2)

        c4 = self.C4(c3)

        c5 = self.C5(c4)
        c5 = self.c5_out_dropout(c5)
        fc1 = self.fc1(c5.view(-1, self.c5_out_flat_shape))
        fc1 = self.fc1_bn(fc1)
        fc2 = self.fc2(F.relu(fc1))
        fc_out = self.fc_out(F.relu(fc2))
        out = fc_out.view(fc_out.shape[0], 2, -1, 1)
        return out


m = SkullNet(1, 2)
torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print('Total Params:', torch_total_params)
