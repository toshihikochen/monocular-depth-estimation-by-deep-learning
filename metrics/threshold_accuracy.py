import torch
import torchmetrics as tm


class ThresholdAccuracy(tm.Metric):
    def __init__(self, threshold=1.25, full_state_update=False):
        super(ThresholdAccuracy, self).__init__(full_state_update=full_state_update)
        self.register_buffer("threshold", torch.tensor(threshold))
        self.add_state("correct", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, target):
        maximum = torch.maximum(preds / target, target / preds)
        self.correct += torch.sum(maximum < self.threshold)
        self.total += torch.numel(maximum)

    def compute(self):
        return self.correct / self.total
