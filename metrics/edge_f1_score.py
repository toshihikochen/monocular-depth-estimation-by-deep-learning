import torch
import torch.nn.functional as F
import torchmetrics as tm


class EdgeF1Score(tm.F1Score):
    def __init__(self, threshold, full_state_update=False):
        super(EdgeF1Score, self).__init__(threshold=threshold, full_state_update=full_state_update)
        self.register_buffer("sobel", torch.tensor([[[
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ]], [[
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]]], dtype=torch.float))

    def update(self, preds, target):
        d_preds = F.conv2d(preds, self.sobel, padding=1)
        preds_edge = torch.sum(d_preds ** 2, dim=1)
        d_target = F.conv2d(target, self.sobel, padding=1)
        target_edge = torch.sqrt(torch.sum(d_target ** 2, dim=1))
        target_edge = torch.where(target_edge > self.threshold, 1, 0)

        super(EdgeF1Score, self).update(preds_edge, target_edge)

    def compute(self):
        return super(EdgeF1Score, self).compute()
