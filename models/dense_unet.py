import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.densenet import densenet169

from models.conv_utils import DoubleConv, Out

class Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super(Encoder, self).__init__()
        self.densenet = densenet169(pretrained=pretrained)
        delattr(self.densenet, 'classifier')

        self.head = nn.Sequential(
            self.densenet.features.conv0,
            self.densenet.features.norm0,
            self.densenet.features.relu0,
        )
        self.pool = self.densenet.features.pool0
        self.layer1_denseblock = self.densenet.features.denseblock1
        self.layer1_transition = self.densenet.features.transition1
        self.layer2_denseblock = self.densenet.features.denseblock2
        self.layer2_transition = self.densenet.features.transition2
        self.layer3_denseblock = self.densenet.features.denseblock3
        self.layer3_transition = self.densenet.features.transition3
        self.layer4_denseblock = self.densenet.features.denseblock4
        self.norm = self.densenet.features.norm5

    def forward(self, x):
        skips = []
        x = self.head(x)
        skips.append(x)
        x = self.pool(x)
        skips.append(x)
        x = self.layer1_denseblock(x)
        x = self.layer1_transition(x)
        skips.append(x)
        x = self.layer2_denseblock(x)
        x = self.layer2_transition(x)
        skips.append(x)
        x = self.layer3_denseblock(x)
        x = self.layer3_transition(x)
        x = self.layer4_denseblock(x)
        x = self.norm(x)
        skips.append(x)
        return skips


class Decoder(nn.Module):
    def __init__(self, norm=False, activation=0., dropout=0.):
        super(Decoder, self).__init__()
        self.conv1 = DoubleConv(1920, 512, kernel_size=3, padding=1, norm=norm, activation=activation, dropout=dropout)
        self.conv2 = DoubleConv(640, 256, kernel_size=3, padding=1, norm=norm, activation=activation, dropout=dropout)
        self.conv3 = DoubleConv(320, 128, kernel_size=3, padding=1, norm=norm, activation=activation, dropout=dropout)
        self.conv4 = DoubleConv(192, 64, kernel_size=3, padding=1, norm=norm, activation=activation, dropout=dropout)
        self.out = Out(64, 1)

    def forward(self, skips):
        x = F.interpolate(skips[4], size=(skips[3].shape[2], skips[3].shape[3]), mode='bilinear', align_corners=True)
        x = torch.cat([x, skips[3]], dim=1)
        x = self.conv1(x)

        x = F.interpolate(x, size=(skips[2].shape[2], skips[2].shape[3]), mode='bilinear', align_corners=True)
        x = torch.cat([x, skips[2]], dim=1)
        x = self.conv2(x)

        x = F.interpolate(x, size=(skips[1].shape[2], skips[1].shape[3]), mode='bilinear', align_corners=True)
        x = torch.cat([x, skips[1]], dim=1)
        x = self.conv3(x)

        x = F.interpolate(x, size=(skips[0].shape[2], skips[0].shape[3]), mode='bilinear', align_corners=True)
        x = torch.cat([x, skips[0]], dim=1)
        x = self.conv4(x)
        x = self.out(x)
        return x


class DenseUNet(nn.Module):
    def __init__(self, pretrained=False, norm=False, activation=0., dropout=0.):
        super(DenseUNet, self).__init__()
        self.encoder = Encoder(pretrained=pretrained)
        self.decoder = Decoder(norm, activation, dropout)

    def forward(self, x):
        skips = self.encoder(x)
        x = self.decoder(skips)
        return x
