import torch
import torch.nn.functional as F
import torchmetrics as tm


class EdgeF1Score(tm.F1Score):
    def __init__(self, threshold):
        super(EdgeF1Score, self).__init__(threshold=threshold)
        self.register_buffer("sobel_x", torch.tensor([[[
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ]]]))
        self.register_buffer("sobel_y", torch.tensor([[[
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]]]))

    def update(self, y_true, y_pred):
        true_x = F.conv2d(y_true, self.sobel_x, padding=1)
        true_y = F.conv2d(y_true, self.sobel_y, padding=1)
        true_edge = torch.sqrt(true_x ** 2 + true_y ** 2)

        pred_x = F.conv2d(y_pred, self.sobel_x, padding=1)
        pred_y = F.conv2d(y_pred, self.sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_x ** 2 + pred_y ** 2)

        super(EdgeF1Score, self).update(true_edge, pred_edge)

    def compute(self):
        return super(EdgeF1Score, self).compute()
