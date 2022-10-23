import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.efficientnet import efficientnet_b0

from models.conv_utils import DoubleConv, Out

class Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super(Encoder, self).__init__()
        self.efficientnet = efficientnet_b0(pretrained=pretrained)
        delattr(self.efficientnet, 'avgpool')
        delattr(self.efficientnet, 'classifier')

        self.block1 = nn.Sequential(
            self.efficientnet.features[0],
            self.efficientnet.features[1],
        )
        self.block2 = nn.Sequential(
            self.efficientnet.features[2],
        )
        self.block3 = nn.Sequential(
            self.efficientnet.features[3],
        )
        self.block4 = nn.Sequential(
            self.efficientnet.features[4],
            self.efficientnet.features[5],
        )
        self.block5 = nn.Sequential(
            self.efficientnet.features[6],
            self.efficientnet.features[7],
        )

    def forward(self, x):
        skips = []
        x = self.block1(x)
        skips.append(x)
        x = self.block2(x)
        skips.append(x)
        x = self.block3(x)
        skips.append(x)
        x = self.block4(x)
        skips.append(x)
        x = self.block5(x)
        skips.append(x)
        return skips


class Decoder(nn.Module):
    def __init__(self, norm=False, activation=0., dropout=0.):
        super(Decoder, self).__init__()
        self.conv1 = DoubleConv(432, 256, kernel_size=3, padding=1, norm=norm, activation=activation, dropout=dropout)
        self.conv2 = DoubleConv(296, 128, kernel_size=3, padding=1, norm=norm, activation=activation, dropout=dropout)
        self.conv3 = DoubleConv(152, 64, kernel_size=3, padding=1, norm=norm, activation=activation, dropout=dropout)
        self.conv4 = DoubleConv(80, 64, kernel_size=3, padding=1, norm=norm, activation=activation, dropout=dropout)
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

        x = F.interpolate(x, size=(skips[-5].shape[2], skips[-5].shape[3]), mode='bilinear', align_corners=True)
        x = torch.cat([x, skips[-5]], dim=1)
        x = self.conv4(x)
        x = self.out(x)
        return x


class EfficientUNet(nn.Module):
    def __init__(self, pretrained=False, norm=False, activation=0., dropout=0.):
        super(EfficientUNet, self).__init__()
        self.encoder = Encoder(pretrained=pretrained)
        self.decoder = Decoder(norm, activation, dropout)

    def forward(self, x):
        skips = self.encoder(x)
        x = self.decoder(skips)
        return x
