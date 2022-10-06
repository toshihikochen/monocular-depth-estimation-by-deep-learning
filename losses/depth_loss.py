import torch
import torch.nn as nn

import torchmetrics.functional as tmf

from losses.gradient_loss import GradientLoss
from losses.ssim_loss import SSIMLoss

class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()
        self.l1loss = nn.L1Loss()
        self.grad_loss = GradientLoss()
        self.ssim_loss = SSIMLoss()

    def forward(self, y_pred, y_true):
        return 0.1*self.l1loss(y_pred, y_true) + self.grad_loss(y_pred, y_true) + self.ssim_loss(y_pred, y_true)
