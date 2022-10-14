import torch
import torch.nn.functional as F
import torchmetrics as tm


class EdgeF1Score(tm.classification.BinaryF1Score):
    higher_is_better = True
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
        # normalize preds and target to [0, 1] range
        maximum = torch.max(torch.max(preds), torch.max(target))
        minimum = torch.min(torch.min(preds), torch.min(target))
        preds = (preds - minimum) / (maximum - minimum)
        target = (target - minimum) / (maximum - minimum)

        d_preds = F.conv2d(preds, self.sobel, padding=1)
        self.preds_edge = torch.sum(d_preds ** 2, dim=1)
        preds_mask = torch.where(self.preds_edge > self.threshold, 1, 0)
        d_target = F.conv2d(target, self.sobel, padding=1)
        self.target_edge = torch.sqrt(torch.sum(d_target ** 2, dim=1))
        target_mask = torch.where(self.target_edge > self.threshold, 1, 0)

        super(EdgeF1Score, self).update(preds_mask, target_mask)
