import torch
import torchmetrics as tm


class StructuralSimilarityIndexMeasure(tm.Metric):
    def __init__(self, full_state_update=False):
        super(StructuralSimilarityIndexMeasure, self).__init__(full_state_update=full_state_update)
        self.add_state("ssim", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("step", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, target):
        self.ssim += tm.functional.structural_similarity_index_measure(preds, target)
        self.step += 1

    def compute(self):
        return self.ssim / self.step
