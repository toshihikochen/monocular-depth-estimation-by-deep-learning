import torch
import torchmetrics as tm


class ThresholdAccuracy(tm.Metric):
    def __init__(self, threshold=1.25):
        super(ThresholdAccuracy, self).__init__()
        self.register_buffer("threshold", torch.tensor(threshold))
        self.add_state("correct", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, y_true, y_pred):
        y_true = torch.clamp_min(y_true, 1e-3)
        y_pred = torch.clamp_min(y_pred, 1e-3)

        sigma = torch.maximum(y_true / y_pred, y_pred / y_true)
        self.correct += torch.sum(sigma < self.threshold)
        self.total += torch.numel(sigma)

    def compute(self):
        return self.correct / self.total
