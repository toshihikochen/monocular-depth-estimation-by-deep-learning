import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet101

from models.doubleconv import DoubleConv, Out


class Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super(Encoder, self).__init__()
        self.resnet = list(resnet101(pretrained=pretrained).children())[:-2]
        self.resnet_block_1 = nn.Sequential(*self.resnet[:3])
        self.resnet_block_2 = nn.Sequential(*self.resnet[3:5])
        self.resnet_block_3 = nn.Sequential(*self.resnet[5:6])
        self.resnet_block_4 = nn.Sequential(*self.resnet[6:])

    def forward(self, x):
        skips = []
        x = self.resnet_block_1(x)
        skips.append(x)
        x = self.resnet_block_2(x)
        skips.append(x)
        x = self.resnet_block_3(x)
        skips.append(x)
        x = self.resnet_block_4(x)
        skips.append(x)
        return skips


class Decoder(nn.Module):
    def __init__(self, norm=False, activation=0., dropout=0.):
        super(Decoder, self).__init__()
        self.conv1 = DoubleConv(2560, 512, kernel_size=3, padding=1, norm=norm, activation=activation, dropout=dropout)
        self.conv2 = DoubleConv(768, 256, kernel_size=3, padding=1, norm=norm, activation=activation, dropout=dropout)
        self.conv3 = DoubleConv(320, 64, kernel_size=3, padding=1, norm=norm, activation=activation, dropout=dropout)
        self.out = Out(64, 1)

    def forward(self, skips):
        x = F.interpolate(skips[-1], size=(skips[-2].shape[2], skips[-2].shape[3]), mode='bilinear', align_corners=True)
        x = torch.cat([x, skips[-2]], dim=1)
        x = self.conv1(x)

        x = F.interpolate(x, size=(skips[-3].shape[2], skips[-3].shape[3]), mode='bilinear', align_corners=True)
        x = torch.cat([x, skips[-3]], dim=1)
        x = self.conv2(x)

        x = F.interpolate(x, size=(skips[-4].shape[2], skips[-4].shape[3]), mode='bilinear', align_corners=True)
        x = torch.cat([x, skips[-4]], dim=1)
        x = self.conv3(x)
        x = self.out(x)
        return x


class ResNetUNet(nn.Module):
    def __init__(self, pretrained=False, norm=False, activation=0., dropout=0.):
        super(ResNetUNet, self).__init__()
        self.encoder = Encoder(pretrained)
        self.decoder = Decoder(norm, activation, dropout)

    def forward(self, x):
        skips = self.encoder(x)
        x = self.decoder(skips)
        return x
