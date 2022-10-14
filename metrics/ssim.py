import torch
import torchmetrics as tm


class StructuralSimilarityIndexMeasure(tm.MeanMetric):
    higher_is_better = True
    def __init__(self, full_state_update=False):
        super(StructuralSimilarityIndexMeasure, self).__init__(full_state_update=full_state_update)

    def update(self, preds, target):
        ssim = tm.functional.structural_similarity_index_measure(preds, target)
        super(StructuralSimilarityIndexMeasure, self).update(ssim)
