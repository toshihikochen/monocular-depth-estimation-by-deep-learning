import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16

from models.conv_utils import DoubleConv, Out


class BottomUp(nn.Module):
    def __init__(self, pretrained=False):
        super(BottomUp, self).__init__()
        self.vgg = list(vgg16(pretrained=pretrained).features.children())[:30]
        self.vgg_block_1 = nn.Sequential(*self.vgg[:9])
        self.vgg_block_2 = nn.Sequential(*self.vgg[9:16])
        self.vgg_block_3 = nn.Sequential(*self.vgg[16:23])
        self.vgg_block_4 = nn.Sequential(*self.vgg[23:])

    def forward(self, x):
        skips = []
        x = self.vgg_block_1(x)
        skips.append(x)
        x = self.vgg_block_2(x)
        skips.append(x)
        x = self.vgg_block_3(x)
        skips.append(x)
        x = self.vgg_block_4(x)
        skips.append(x)
        return skips

class TopDown(nn.Module):
    def __init__(self):
        super(TopDown, self).__init__()
        self.connection1 = nn.Conv2d(512, 128, kernel_size=1)
        self.connection2 = nn.Conv2d(512, 128, kernel_size=1)
        self.connection3 = nn.Conv2d(256, 128, kernel_size=1)
        self.connection4 = nn.Conv2d(128, 128, kernel_size=1)
        self.out1 = Out(128, 1)
        self.out2 = Out(128, 1)
        self.out3 = Out(128, 1)
        self.out4 = Out(128, 1)

    def forward(self, skips):
        x = self.connection1(skips[3])
        x1 = self.out1(x)

        x = F.interpolate(x, size=(skips[2].shape[2], skips[2].shape[3]), mode='nearest')
        x = x + self.connection2(skips[2])
        x2 = self.out2(x)

        x = F.interpolate(x, size=(skips[1].shape[2], skips[1].shape[3]), mode='nearest')
        x = x + self.connection3(skips[1])
        x3 = self.out3(x)

        x = F.interpolate(x, size=(skips[0].shape[2], skips[0].shape[3]), mode='nearest')
        x = x + self.connection4(skips[0])
        x4 = self.out4(x)

        return x1, x2, x3, x4


class TopDownSingle(nn.Module):
    def __init__(self):
        super(TopDownSingle, self).__init__()
        self.connection1 = nn.Conv2d(512, 128, kernel_size=1)
        self.connection2 = nn.Conv2d(512, 128, kernel_size=1)
        self.connection3 = nn.Conv2d(256, 128, kernel_size=1)
        self.connection4 = nn.Conv2d(128, 128, kernel_size=1)
        self.out = Out(128, 1)

    def forward(self, skips):
        x = self.connection1(skips[3])

        x = F.interpolate(x, size=(skips[2].shape[2], skips[2].shape[3]), mode='nearest')
        x = x + self.connection2(skips[2])

        x = F.interpolate(x, size=(skips[1].shape[2], skips[1].shape[3]), mode='nearest')
        x = x + self.connection3(skips[1])

        x = F.interpolate(x, size=(skips[0].shape[2], skips[0].shape[3]), mode='nearest')
        x = x + self.connection4(skips[0])

        x = self.out(x)
        return x


class VGGFPN(nn.Module):
    num_feature_maps = 4
    def __init__(self, pretrained=False, single=True):
        super(VGGFPN, self).__init__()
        self.bottom_up = BottomUp(pretrained=pretrained)
        self.top_down = TopDownSingle() if single else TopDown()

    def forward(self, x):
        skips = self.bottom_up(x)
        x = self.top_down(skips)
        return x
