import torch
import torchmetrics as tm


class ThresholdAccuracy(tm.MeanMetric):
    higher_is_better = True
    def __init__(self, threshold=1.25, full_state_update=False):
        super(ThresholdAccuracy, self).__init__(full_state_update=full_state_update)
        self.register_buffer("threshold", torch.tensor(threshold))

    def update(self, preds, target):
        self.maximum = torch.maximum(preds / target, target / preds)
        mask = torch.where(self.maximum < self.threshold, 1, 0)
        super(ThresholdAccuracy, self).update(mask)
