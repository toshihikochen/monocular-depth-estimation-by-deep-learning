import torch
import torchmetrics as tm


class Log10AverageError(tm.Metric):
    def __init__(self):
        super(Log10AverageError, self).__init__()
        self.add_state("log_error", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("step", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, y_true, y_pred):
        y_true = torch.clamp_min(y_true, 1e-3)
        y_pred = torch.clamp_min(y_pred, 1e-3)
        self.log_error += torch.mean(torch.abs(torch.log10(y_true) - torch.log10(y_pred)))
        self.step += 1

    def compute(self):
        return self.log_error / self.step
