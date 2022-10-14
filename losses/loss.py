import torch
import torch.nn as nn

from losses import DepthLoss, GradientLoss, SSIMLoss


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.depth_loss = DepthLoss()
        self.gradient_loss = GradientLoss()
        self.ssim_loss = SSIMLoss()

    def forward(self, y_pred, y_true):
        return self.depth_loss(y_pred, y_true) + \
               self.gradient_loss(y_pred, y_true) + \
               self.ssim_loss(y_pred, y_true) / 2
