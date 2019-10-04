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
    def __init__(self, in_channels, middle_channel, out_channels, p=0):
        super(_DoubleConvolution, self).__init__()
        layers = [
            nn.Conv2d(in_channels, middle_channel, kernel_size=3, padding=p),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channel, out_channels, kernel_size=3, padding=p),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class UNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(UNet, self).__init__()

        reduce_by = 4

        self.A2_ = _DoubleConvolution(num_channels, int(128 / reduce_by), int(128 / reduce_by))
        self.A3_ = _DoubleConvolution(int(128 / reduce_by), int(256 / reduce_by), int(256 / reduce_by))
        self.A4_ = _DoubleConvolution(int(256 / reduce_by), int(512 / reduce_by), int(512 / reduce_by))

        self.A_mid = _DoubleConvolution(int(512 / reduce_by), int(1024 / reduce_by), int(1024 / reduce_by))

        self.A4_up = nn.ConvTranspose2d(int(1024 / reduce_by), int(512 / reduce_by), kernel_size=2, stride=2)
        self._A4 = _DoubleConvolution(int(1024 / reduce_by), int(512 / reduce_by), int(512 / reduce_by))

        self.A3_up = nn.ConvTranspose2d(int(512 / reduce_by), int(256 / reduce_by), kernel_size=2, stride=2)
        self._A3 = _DoubleConvolution(int(512 / reduce_by), int(256 / reduce_by), int(256 / reduce_by))

        self.A2_up = nn.ConvTranspose2d(int(256 / reduce_by), int(128 / reduce_by), kernel_size=2, stride=2)
        self._A2 = _DoubleConvolution(int(256 / reduce_by), int(128 / reduce_by), int(128 / reduce_by))

        self.out1 = nn.Conv2d(int(128 / reduce_by), int(64 / reduce_by), kernel_size=3, padding=0, stride=2)
        self.out2 = nn.Conv2d(int(64 / reduce_by), num_classes, kernel_size=3, padding=0, stride=2)
        self.fc1 = nn.Linear(2 * 48 * 48, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_out = nn.Linear(256, 12)
        initialize_model_weights(self)
        self.num_classes = num_classes

    def forward(self, x):
        a2_ = self.A2_(x)
        a2_dwn = F.max_pool2d(a2_, kernel_size=2, stride=2)

        a3_ = self.A3_(a2_dwn)
        a3_dwn = F.max_pool2d(a3_, kernel_size=2, stride=2)

        a4_ = self.A4_(a3_dwn)
        # a4_ = F.dropout(a4_, p=0.2)
        a4_dwn = F.max_pool2d(a4_, kernel_size=2, stride=2)

        a_mid = self.A_mid(a4_dwn)

        a4_up = self.A4_up(a_mid)
        _a4 = self._A4(UNet.match_and_concat(a4_, a4_up))
        # _a4 = F.dropout(_a4, p=0.2)

        a3_up = self.A3_up(_a4)
        _a3 = self._A3(UNet.match_and_concat(a3_, a3_up))

        a2_up = self.A2_up(_a3)
        _a2 = self._A2(UNet.match_and_concat(a2_, a2_up))

        out1 = F.relu(self.out1(_a2))
        out2 = F.relu(self.out2(out1))
        fc1 = self.fc1(out2.view(-1, 2 * 48 * 48))
        fc1 = F.dropout(F.relu(fc1), 0.2)
        fc2 = F.relu(self.fc2(fc1))
        fc_out = self.fc_out(fc2)
        final = fc_out.view(fc_out.shape[0], self.num_classes, -1)
        return final

    @staticmethod
    def match_and_concat(bypass, upsampled, crop=True):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)


m = UNet(1, 2)
torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print('Total Params:', torch_total_params)


class SkullNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(SkullNet, self).__init__()
        self.reduce_by = 2
        self.num_classes = num_classes

        self.C1 = _DoubleConvolution(num_channels, int(64 / self.reduce_by), int(64 / self.reduce_by))
        self.C2 = _DoubleConvolution(int(64 / self.reduce_by), int(128 / self.reduce_by), int(128 / self.reduce_by))
        self.C3 = _DoubleConvolution(int(128 / self.reduce_by), int(256 / self.reduce_by), int(256 / self.reduce_by))
        self.C4 = _DoubleConvolution(int(256 / self.reduce_by), int(512 / self.reduce_by), int(256 / self.reduce_by))
        self.C5 = _DoubleConvolution(int(256 / self.reduce_by), int(128 / self.reduce_by), int(64 / self.reduce_by))
        self.fc1 = nn.Linear(32 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_out = nn.Linear(256, 12)
        initialize_model_weights(self)

    def forward(self, x):
        c1 = self.C1(x)
        c1_mxp = F.max_pool2d(c1, kernel_size=2, stride=2)

        c2 = self.C2(c1_mxp)
        c2_mxp = F.max_pool2d(c2, kernel_size=2, stride=2)

        c3 = self.C3(c2_mxp)
        c3_mxp = F.max_pool2d(c3, kernel_size=2, stride=2)

        c4 = self.C4(c3_mxp)
        c4_mxp = F.max_pool2d(c4, kernel_size=2, stride=2)

        c5 = self.C5(c4_mxp)

        fc1 = self.fc1(c5.view(-1, 32 * 8 * 8))
        fc1 = F.dropout(fc1, 0.2)
        fc2 = self.fc2(F.relu(fc1))
        fc_out = self.fc_out(F.relu(fc2))
        out = fc_out.view(fc_out.shape[0], 2, -1)
        return out

    @staticmethod
    def match_and_concat(bypass, upsampled, crop=True):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)
