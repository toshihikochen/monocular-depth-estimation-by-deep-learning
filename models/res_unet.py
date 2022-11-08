from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet101

from models.conv_utils import DoubleConv, Out


class Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super(Encoder, self).__init__()
        self.resnet = list(resnet101(pretrained=pretrained).children())[:-2]
        self.resnet_block_1 = nn.Sequential(*self.resnet[:3])
        self.resnet_block_2 = nn.Sequential(*self.resnet[3:5])
        self.resnet_block_3 = nn.Sequential(*self.resnet[5:6])
        self.resnet_block_4 = nn.Sequential(*self.resnet[6:7])
        self.resnet_block_5 = nn.Sequential(*self.resnet[7:])

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
        x = self.resnet_block_5(x)
        skips.append(x)
        return skips


class Decoder(nn.Module):
    def __init__(self, norm=False, activation=0., dropout=0.):
        super(Decoder, self).__init__()
        self.conv1 = DoubleConv(3072, 512, kernel_size=3, padding=1, norm=norm, activation=activation, dropout=dropout)
        self.conv2 = DoubleConv(1024, 256, kernel_size=3, padding=1, norm=norm, activation=activation, dropout=dropout)
        self.conv3 = DoubleConv(512, 128, kernel_size=3, padding=1, norm=norm, activation=activation, dropout=dropout)
        self.conv4 = DoubleConv(192, 64, kernel_size=3, padding=1, norm=norm, activation=activation, dropout=dropout)
        self.out = Out(64, 1)

    def forward(self, skips: List[torch.Tensor]):
        x = F.interpolate(skips[-1], size=(skips[-2].shape[2], skips[-2].shape[3]), mode='bilinear', align_corners=True)
        x = torch.cat([x, skips[-2]], dim=1)
        x = self.conv1(x)

        x = F.interpolate(x, size=(skips[-3].shape[2], skips[-3].shape[3]), mode='bilinear', align_corners=True)
        x = torch.cat([x, skips[-3]], dim=1)
        x = self.conv2(x)

        x = F.interpolate(x, size=(skips[-4].shape[2], skips[-4].shape[3]), mode='bilinear', align_corners=True)
        x = torch.cat([x, skips[-4]], dim=1)
        x = self.conv3(x)

        x = F.interpolate(x, size=(skips[-5].shape[2], skips[-5].shape[3]), mode='bilinear', align_corners=True)
        x = torch.cat([x, skips[-5]], dim=1)
        x = self.conv4(x)

        x = self.out(x)
        return x


class ResUNet(nn.Module):
    def __init__(self, pretrained=False, norm=False, activation=0., dropout=0.):
        super(ResUNet, self).__init__()
        self.encoder = Encoder(pretrained)
        self.decoder = Decoder(norm, activation, dropout)

    def forward(self, x):
        skips = self.encoder(x)
        x = self.decoder(skips)
        return x
