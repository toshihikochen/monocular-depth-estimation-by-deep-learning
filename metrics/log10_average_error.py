import torch
import torchmetrics as tm


class Log10AverageError(tm.MeanAbsoluteError):
    def __init__(self):
        super(Log10AverageError, self).__init__()

    def update(self, y_true, y_pred):
        y_true = torch.clamp_min(y_true, 1e-3)
        y_pred = torch.clamp_min(y_pred, 1e-3)
        super(Log10AverageError, self).update(torch.log10(y_true), torch.log10(y_pred))

    def compute(self):
        return super(Log10AverageError, self).compute()
