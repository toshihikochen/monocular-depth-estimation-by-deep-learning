import torch
import torch.nn as nn

import torchmetrics.functional as tmf


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, y_pred, y_true):
        dy_pred, dx_pred = tmf.image_gradients(y_pred)
        dy_true, dx_true = tmf.image_gradients(y_true)
        return torch.mean(torch.abs(dy_pred - dy_true)) + torch.mean(torch.abs(dx_pred - dx_true))
