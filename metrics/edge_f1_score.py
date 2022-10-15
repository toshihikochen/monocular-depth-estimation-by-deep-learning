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

    def _sobel(self, img: torch.Tensor) -> torch.Tensor:
        d_img = F.conv2d(img, self.sobel, padding=1)
        return torch.sqrt(torch.sum(d_img ** 2, dim=1))

    def update(self, preds, target):
        # normalize preds and target to [0, 1] range
        maximum = torch.max(torch.max(preds), torch.max(target))
        minimum = torch.min(torch.min(preds), torch.min(target))
        preds = (preds - minimum) / (maximum - minimum)
        target = (target - minimum) / (maximum - minimum)

        preds_edge = self._sobel(preds)
        preds_mask = torch.where(preds_edge > self.threshold, 1, 0)
        target_edge = self._sobel(target)
        target_mask = torch.where(target_edge > self.threshold, 1, 0)

        super(EdgeF1Score, self).update(preds_mask, target_mask)
