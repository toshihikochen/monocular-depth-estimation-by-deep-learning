import torch
import torch.nn as nn
import torch.nn.functional as F

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


class MultiScaleLoss(nn.Module):
    def __init__(self, num_scale=2, weights=None):
        super(MultiScaleLoss, self).__init__()
        self.num_scale = num_scale
        self.weights = [1.] * num_scale if weights is None else weights
        self.loss = Loss()

    def forward(self, y_pred, y_true):
        loss = 0.
        for i in range(self.num_scale):
            _y_true = F.interpolate(y_true, size=(y_pred[i].shape[2], y_true[i].shape[3]), mode='nearest')
            loss += self.loss(y_pred[i], _y_true) * self.weights[i]
        return loss
