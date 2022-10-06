import torch
import torchmetrics as tm


class StructuralSimilarityIndexMeasure(tm.Metric):
    def __init__(self):
        super(StructuralSimilarityIndexMeasure, self).__init__()
        self.add_state("ssim", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("step", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, sr_image, hr_image):
        self.ssim += torch.clamp(tm.functional.structural_similarity_index_measure(sr_image, hr_image), 0, 1)
        self.step += 1

    def compute(self):
        return self.ssim / self.step
