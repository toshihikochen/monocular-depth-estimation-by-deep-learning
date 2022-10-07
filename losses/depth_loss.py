import math

import torch
import torch.nn as nn


class DepthLoss(nn.Module):
    def __init__(self, alpha=math.e):
        super(DepthLoss, self).__init__()
        self.register_buffer("alpha", torch.tensor(alpha))

    def forward(self, y_pred, y_true):
        return torch.mean(torch.log(torch.abs(y_pred - y_true) + self.alpha))
