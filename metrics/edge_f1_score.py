import torch
import torch.nn.functional as F
import torchmetrics as tm


class EdgeF1Score(tm.F1Score):
    def __init__(self, threshold):
        super(EdgeF1Score, self).__init__(threshold=threshold)
        self.register_buffer("sobel", torch.tensor([[[
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ]], [[
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]]], dtype=torch.float))

    def update(self, y_true, y_pred):
        d_true = F.conv2d(y_true, self.sobel, padding=1)
        true_edge = torch.sum(d_true ** 2, dim=1, keepdim=True)
        true_edge = torch.where(true_edge > self.threshold, 1, 0)
        d_pred = F.conv2d(y_pred, self.sobel, padding=1)
        pred_edge = torch.sqrt(torch.sum(d_pred ** 2, dim=1, keepdim=True))

        super(EdgeF1Score, self).update(pred_edge, true_edge)

    def compute(self):
        return super(EdgeF1Score, self).compute()
