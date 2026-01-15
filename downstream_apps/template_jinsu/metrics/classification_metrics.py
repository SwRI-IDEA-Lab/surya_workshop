import torch
import torch.nn as nn

# from torch.distributed import all_reduce, ReduceOp
# from surya.utils.distributed import is_dist_avail_and_initialized


import torch
import torch.nn as nn


class DistributedClassificationMetrics(nn.Module):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.register_buffer("_counts", torch.zeros(4))  # tp, tn, fp, fn

    @property
    def tp(self):
        return self._counts[0]

    @property
    def tn(self):
        return self._counts[1]

    @property
    def fp(self):
        return self._counts[2]

    @property
    def fn(self):
        return self._counts[3]

    def update(self, prediction, target):
        prediction = (prediction > self.threshold).int()
        target = target.int()

        self._counts[0] += ((prediction == 1) & (target == 1)).sum()
        self._counts[1] += ((prediction == 0) & (target == 0)).sum()
        self._counts[2] += ((prediction == 1) & (target == 0)).sum()
        self._counts[3] += ((prediction == 0) & (target == 1)).sum()

    def compute_and_reset(self):
        result = {
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
        }
        result["accuracy"] = (self.tp + self.tn) / (
            self.tp + self.tn + self.fp + self.fn
        )
        result["precision"] = self.tp / (self.tp + self.fp)
        result["recall"] = self.tp / (self.tp + self.fn)
        result["f1"] = 2 * self.tp / (2 * self.tp + self.fp + self.fn)
        result["tss"] = self.tp / (self.tp + self.fn) - self.fp / (self.fp + self.tn)

        n = self.tn + self.fp
        p = self.tp + self.fn
        result["hss"] = (
            2.0
            * (self.tp * self.tn - self.fn * self.fp)
            / (p * (self.fn + self.tn) + (self.tp + self.fp) * n)
        )

        self.reset()

        return result

    def reset(self):
        self._counts.zero_()
