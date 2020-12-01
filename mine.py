from typing import Callable
from collections import Counter
import random
import numpy as np
from sampler_abc import BaseSampler
from utility import compute_f_score


class MySampler(BaseSampler):
    def __init__(self, alpha: float,
                 predictions,  # 1d numpy array
                 scores,  # 1d numpy array
                 oracle: Callable,
                 n_bins=30,
                 max_iter=None):
        super().__init__(alpha, predictions, scores, oracle, max_iter=max_iter)
        self.TP, self.FP, self.FN = [0] * 3
        bins = [x/n_bins for x in range(n_bins+1)]
        # print(bins)
        self.strata_allocations = np.digitize(self.scores, bins, right=True)
        strata_counts = Counter(self.strata_allocations)
        self.n_totals = len(scores)
        strata_proportions = {k: v / self.n_totals for k, v in strata_counts.items()}
        self.weights = {k: v/(1/n_bins) for k, v in strata_proportions.items()}

    def select_next_item(self, sample_with_replacement: bool, **kwargs):
        strata_idx = random.choice(list(self.weights.keys()))
        loc = np.random.choice(np.arange(self.n_totals)[self.strata_allocations == strata_idx])
        return loc, self.weights[strata_idx], None

    def update_estimate_and_sampler(self, ell, ell_hat, weight, **kwargs):
        """Update the estimate after querying the label for an item"""
        self.TP += ell_hat * ell * weight
        self.FP += ell_hat * (1 - ell) * weight
        self.FN += (1 - ell_hat) * ell * weight
        self.f_scores[self.idx] = compute_f_score(self.alpha, self.TP, self.FP, self.FN)




