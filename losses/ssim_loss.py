import torch
import torch.nn as nn

import torchmetrics.functional as tmf


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.clamp((1 - tmf.structural_similarity_index_measure(y_pred, y_true)) / 2, 0, 1)
