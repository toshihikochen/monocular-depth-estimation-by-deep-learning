import torch
import torch.nn as nn


def _gradients(img: torch.Tensor, stride=1, padding=1):

    batch_size, channels, height, width = img.shape

    dy = img[..., stride:, :] - img[..., :-stride, :]
    dx = img[..., :, stride:] - img[..., :, :-stride]

    shapey = [batch_size, channels, padding, width]
    dy = torch.cat([dy, torch.zeros(shapey, device=img.device, dtype=img.dtype)], dim=2)

    shapex = [batch_size, channels, height, padding]
    dx = torch.cat([dx, torch.zeros(shapex, device=img.device, dtype=img.dtype)], dim=3)

    return dy, dx


class GradientLoss(nn.Module):
    def __init__(self, stride=1, padding=None):
        super(GradientLoss, self).__init__()
        self.stride = stride
        self.padding = stride if padding is None else padding

    def forward(self, y_pred, y_true):
        dy_pred, dx_pred = _gradients(y_pred, self.stride, self.padding)
        dy_true, dx_true = _gradients(y_true, self.stride, self.padding)
        return torch.mean(torch.abs(dy_pred - dy_true)) + torch.mean(torch.abs(dx_pred - dx_true))
