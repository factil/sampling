from typing import Callable, Tuple
import numpy as np
import warnings

from sampler_abc import BaseSampler
from utility import compute_f_score


class PassiveSampler(BaseSampler):
    def __init__(self, alpha: float,
                 predictions, # 1d numpy array
                 scores, # 1d numpy array
                 oracle: Callable,
                 max_iter=None):

        super().__init__(alpha, predictions, scores, oracle, max_iter=max_iter)
        self.TP, self.FP, self.FN = [0] * 3

    def select_next_item(self, sample_with_replacement: bool, **kwargs) -> Tuple:
        """Sample an item from the pool"""
        if sample_with_replacement:
            # Can sample from any of the items
            loc = np.random.choice(self.n_items)
        else:
            # Can only sample from items that have not been seen
            # Find ids that haven't been seen yet
            not_seen_ids, = np.where(np.isnan(self.cached_labels))
            if len(not_seen_ids) == 0:
                raise ValueError("all have been sampled")
            loc = np.random.choice(not_seen_ids)
        return loc, 1, None

    def update_estimate_and_sampler(self, ell, ell_hat, weight, **kwargs):
        """Update the estimate after querying the label for an item"""
        self.TP += ell_hat * ell * weight
        self.FP += ell_hat * (1 - ell) * weight
        self.FN += (1 - ell_hat) * ell * weight
        self.f_scores[self.idx] = compute_f_score(self.alpha, self.TP, self.FP, self.FN)
