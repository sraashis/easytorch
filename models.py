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

        self.B1_, self.B2_, self.B3_, self._B1, self._B2, self._B3, self.B = \
            ConvDown2D(num_channels, 64), ConvDown2D(64, 64), ConvDown2D(64, 64), \
            ConvUp2D(64, 64), \
            ConvUp2D(64, 64), ConvUp2D(64, 64), Conv2D(64, 64)

        self.C1_, self.C2_, self.C3_, self._C1, self._C2, self._C3, self.C = \
            ConvDown2D(64, 256), ConvDown2D(320, 256), ConvDown2D(320, 256), \
            ConvUp2D(320, 256), \
            ConvUp2D(320, 256), ConvUp2D(320, 256), Conv2D(320, 256)

        self.D1_, self.D2_, self.D3_, self._D1, self._D2, self._D3, self.D = \
            ConvDown2D(256, 64), ConvDown2D(320, 64), ConvDown2D(320, 64), \
            ConvUp2D(320, 64), \
            ConvUp2D(320, 64), ConvUp2D(320, 64), Conv2D(320, 64)

        self.O1, self.O2, self.O3, self.O4, self.O5, self.O6, self.O7 = \
            ConvOut(64, num_classes), ConvOut(64, num_classes), ConvOut(64, num_classes), \
            ConvOut(64, num_classes), \
            ConvOut(64, num_classes), ConvOut(64, num_classes), ConvOut(64, num_classes)
        initialize_weights(self)

    def forward(self, x):
        B1_, B1_dwn = self.B1_(x)
        B2_, B2_dwn = self.B2_(B1_dwn)
        B3_, B3_dwn = self.B3_(B2_dwn)
        _B1, B1_up = self._B1(B3_dwn)
        _B2, B2_up = self._B2(B1_up)
        _B3, B3_up = self._B3(B2_up)
        B = self.B(B3_up)

        C1_, C1_dwn = self.C1_(B1_)
        C2_, C2_dwn = self.C2_(cat([C1_dwn, B2_], 1))
        C3_, C3_dwn = self.C3_(cat([C2_dwn, B3_], 1))
        _C1, C1_up = self._C1(cat([C3_dwn, _B1], 1))
        _C2, C2_up = self._C2(cat([C1_up, _B2], 1))
        _C3, C3_up = self._C3(cat([C2_up, _B3], 1))
        C = self.C(cat([C3_up, B], 1))

        D1_, D1_dwn = self.D1_(C1_)
        D2_, D2_dwn = self.D2_(cat([D1_dwn, C2_], 1))
        D3_, D3_dwn = self.D3_(cat([D2_dwn, C3_], 1))
        _D1, D1_up = self._D1(cat([D3_dwn, _C1], 1))
        _D2, D2_up = self._D2(cat([D1_up, _C2], 1))
        _D3, D3_up = self._D3(cat([D2_up, _C3], 1))
        D = self.D(cat([D3_up, C], 1))

        return self.O1(D1_), self.O2(D2_), self.O3(D3_), self.O4(_D1), self.O5(_D2), self.O6(_D3), self.O7(D)


m = FishNet(1, 2)
torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print('Total Params:', torch_total_params)
