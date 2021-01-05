from typing import Callable
from collections import Counter
import random
import numpy as np
from .sampler_abc import SamplerInternal
from .utility import compute_f_score


class StratifiedUniformSampler(SamplerInternal):
    def __init__(self,
                 alpha: float,
                 scores,  # 1d numpy array
                 strata,
                 threshold=0.5, **kwargs):
        super().__init__(alpha, scores, strata, threshold=threshold, **kwargs)
        self.strata = strata
        self.n_strata = len(self.strata)
        self.TP, self.FP, self.FN = [0] * 3
        self.n_totals = len(scores)
        self.weights = [(v/self.n_totals) / (1/self.n_strata) for v in self.strata.sizes]
        self.current_weight = None

    def _select(self) -> int:
        """
        select return a strata idx
        means if we have a labelled pair with a score that falls in a strata
        we can reuse a old labelled pair instead of consulting human oracle.
        """
        stratum_idx = np.random.choice(np.arange(self.n_strata)[self.strata.sizes > 0])
        self.current_weight = self.weights[stratum_idx]
        return stratum_idx

    def _set(self, idx: int, label: bool):
        """
        set label for the pair selected
        """
        ell = label
        ell_hat = self.predictions[idx]
        weight = self.current_weight
        self.TP += ell_hat * ell * weight
        self.FP += ell_hat * (1 - ell) * weight
        self.FN += (1 - ell_hat) * ell * weight
        self.f_scores.append(compute_f_score(self.alpha, self.TP, self.FP, self.FN))



