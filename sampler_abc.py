import random
from abc import ABC, abstractmethod
import numpy as np

from utility import is_between_zero_and_one


class SamplerInternal(ABC):
    def __init__(self,
                 alpha: float,
                 scores,
                 strata,
                 threshold=0.5, **kwargs):
        if not is_between_zero_and_one(alpha):
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha

        if not is_between_zero_and_one(scores):
            raise ValueError("scores must be between 0 and 1")
        self.scores = scores
        self.pairs_sampled = []
        self.predictions = (self.scores > threshold).astype(np.float)
        self.f_scores = []
        self.is_select_call = True

    @abstractmethod
    def _select(self) -> int:
        """
        select return a strata idx
        means if we have a labelled pair with a score that falls in a strata
        we can reuse a old labelled pair instead of consulting human oracle.
        """
        pass

    @abstractmethod
    def _set(self, idx: int, label: bool):
        """
        set label for the pair selected
        """
        pass

    def select(self) -> int:  # strata idx
        if not self.is_select_call:
            raise ValueError("next call should be `next`")
        result = self._select()
        self.is_select_call = False
        return result

    def set(self, idx: int, label: bool):
        if self.is_select_call:
            raise ValueError("next call should be `select`")
        self._set(idx, label)
        self.is_select_call = True
        self.pairs_sampled.append((idx, label))

    def f_score_history(self):
        return self.f_scores

    def pairs(self):
        return self.pairs_sampled



