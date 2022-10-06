import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,
                 norm=False, activation=0., dropout=0.):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.norm1 = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.activation1 = nn.Identity() if activation is False else \
            nn.ReLU(inplace=True) if activation == 0. else \
            nn.LeakyReLU(activation, inplace=True)
        self.dropout1 = nn.Dropout(p=dropout) if dropout else nn.Identity()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.norm2 = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.activation2 = nn.Identity() if activation is False else \
            nn.ReLU(inplace=True) if activation == 0. else \
                nn.LeakyReLU(activation, inplace=True)
        self.dropout2 = nn.Dropout(p=dropout) if dropout else nn.Identity()

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        return x


class Out(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Out, self).__init__()
        self.out = nn.Conv2d(in_channels, out_channels=num_classes, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = self.out(x)
        return x
