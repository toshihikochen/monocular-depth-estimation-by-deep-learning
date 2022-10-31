import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.efficientnet import efficientnet_b0

from models.conv_utils import Out


class BottomUp(nn.Module):
    def __init__(self, pretrained=False):
        super(BottomUp, self).__init__()
        self.efficientnet = efficientnet_b0(pretrained=pretrained)
        delattr(self.efficientnet, 'avgpool')
        delattr(self.efficientnet, 'classifier')
        # replace all swish activations with relu6 for quantization
        for name, module in self.efficientnet.named_modules():
            if isinstance(module, nn.SiLU):
                setattr(self.efficientnet, name, nn.ReLU6(inplace=True))

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


class TopDown(nn.Module):
    def __init__(self):
        super(TopDown, self).__init__()
        self.connection1 = nn.Conv2d(320, 16, kernel_size=1)
        self.connection2 = nn.Conv2d(112, 16, kernel_size=1)
        self.connection3 = nn.Conv2d(40, 16, kernel_size=1)
        self.connection4 = nn.Conv2d(24, 16, kernel_size=1)
        self.connection5 = nn.Conv2d(16, 16, kernel_size=1)
        self.out1 = Out(16, 1)
        self.out2 = Out(16, 1)
        self.out3 = Out(16, 1)
        self.out4 = Out(16, 1)
        self.out5 = Out(16, 1)

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
        self.connection1 = nn.Conv2d(320, 16, kernel_size=1)
        self.connection2 = nn.Conv2d(112, 16, kernel_size=1)
        self.connection3 = nn.Conv2d(40, 16, kernel_size=1)
        self.connection4 = nn.Conv2d(24, 16, kernel_size=1)
        self.connection5 = nn.Conv2d(16, 16, kernel_size=1)
        self.out = Out(16, 1)

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


class EfficientFPN(nn.Module):
    num_feature_maps = 5
    def __init__(self, pretrained=False, single=True):
        super(EfficientFPN, self).__init__()
        self.bottom_up = BottomUp(pretrained=pretrained)
        self.top_down = TopDownSingle() if single else TopDown()

    def forward(self, x):
        skips = self.bottom_up(x)
        x = self.top_down(skips)
        return x
