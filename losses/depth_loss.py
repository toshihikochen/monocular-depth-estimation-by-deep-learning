import math

import torch
import torch.nn as nn


class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()
        self.l1loss = nn.L1Loss()

    def forward(self, y_pred, y_true):
        return self.l1loss(y_pred, y_true)
