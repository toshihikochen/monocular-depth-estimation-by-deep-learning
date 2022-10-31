import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.densenet import densenet169

from models.conv_utils import Out


class BottomUp(nn.Module):
    def __init__(self, pretrained=False):
        super(BottomUp, self).__init__()
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


class TopDown(nn.Module):
    def __init__(self):
        super(TopDown, self).__init__()
        self.connection1 = nn.Conv2d(1664, 64, kernel_size=1)
        self.connection2 = nn.Conv2d(256, 64, kernel_size=1)
        self.connection3 = nn.Conv2d(128, 64, kernel_size=1)
        self.connection4 = nn.Conv2d(64, 64, kernel_size=1)
        self.connection5 = nn.Conv2d(64, 64, kernel_size=1)
        self.out1 = Out(64, 1)
        self.out2 = Out(64, 1)
        self.out3 = Out(64, 1)
        self.out4 = Out(64, 1)
        self.out5 = Out(64, 1)

    def forward(self, skips):
        x = self.connection1(skips[4])
        x1 = self.out1(x)

        x = F.interpolate(x, size=(skips[3].shape[2], skips[3].shape[3]), mode='nearest')
        x = x + self.connection2(skips[3])
        x2 = self.out2(x)

        x = F.interpolate(x, size=(skips[2].shape[2], skips[2].shape[3]), mode='nearest')
        x = x + self.connection3(skips[2])
        x3 = self.out3(x)

        x = F.interpolate(x, size=(skips[1].shape[2], skips[1].shape[3]), mode='nearest')
        x = x + self.connection4(skips[1])
        x4 = self.out4(x)

        x = F.interpolate(x, size=(skips[0].shape[2], skips[0].shape[3]), mode='nearest')
        x = x + self.connection5(skips[0])
        x5 = self.out5(x)

        return x1, x2, x3, x4, x5


class TopDownSingle(nn.Module):
    def __init__(self):
        super(TopDownSingle, self).__init__()
        self.connection1 = nn.Conv2d(1664, 64, kernel_size=1)
        self.connection2 = nn.Conv2d(256, 64, kernel_size=1)
        self.connection3 = nn.Conv2d(128, 64, kernel_size=1)
        self.connection4 = nn.Conv2d(64, 64, kernel_size=1)
        self.connection5 = nn.Conv2d(64, 64, kernel_size=1)
        self.out = Out(64, 1)

    def forward(self, skips):
        x = self.connection1(skips[4])
        x = F.interpolate(x, size=(skips[3].shape[2], skips[3].shape[3]), mode='nearest')
        x = x + self.connection2(skips[3])

        x = F.interpolate(x, size=(skips[2].shape[2], skips[2].shape[3]), mode='nearest')
        x = x + self.connection3(skips[2])

        x = F.interpolate(x, size=(skips[1].shape[2], skips[1].shape[3]), mode='nearest')
        x = x + self.connection4(skips[1])

        x = F.interpolate(x, size=(skips[0].shape[2], skips[0].shape[3]), mode='nearest')
        x = x + self.connection5(skips[0])

        x = self.out(x)
        return x


class DenseFPN(nn.Module):
    num_feature_maps = 5
    def __init__(self, pretrained=False, single=True):
        super(DenseFPN, self).__init__()
        self.bottom_up = BottomUp(pretrained=pretrained)
        self.top_down = TopDownSingle() if single else TopDown()

    def forward(self, x):
        skips = self.bottom_up(x)
        x = self.top_down(skips)
        return x
