import random
from typing import List
from abc import ABC, abstractmethod
import numpy as np

from utility import is_between_zero_and_one


class SamplerInternal(ABC):
    """
    SamplerInternal abstracts stratified sampling. it provides user with a simple interface. this approach
    fits with the bayesian sampling paradigm.

    it also allow the reuse of labelled pairs, as instance of this class will only ask for a particular
    strata instead of a particular pair when sampling. so it gives another label providing class to return
    a labelled pair or consult a human oracle for answer.

    select: decides which strata it would like to sample next

    set: get a pair from that strata and its label and use it to update its internal parameters (if a bayesian model
    is available)

    each select call must be followed by a set call. (enforced), which also is considered as one iteration

    it keeps F score history can be obtained by .f_score_history

    it keeps the index of pairs sampled at each iteration, can be used for record keeping
    """
    def __init__(self,
                 alpha: float,
                 scores,
                 strata,
                 threshold=0.5, **kwargs):
        """
        Parameters
        ----------
        alpha: float between 0 and 1, determine which f score we are estimating
        scores: a array of floats between 0 and 1
        strata: instance of Strata class
        threshold: float between 0 and 1, values above the threshold are true, while the ones
        below it false.

        kwargs
        """
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
        needs to be overridden
        Returns a strata index
        -------

        """
        pass

    @abstractmethod
    def _set(self, idx: int, label: bool):
        """
        set label for the pair selected
        params:
            idx: index of the pair in the array
            label: 0 or 1
        """
        pass

    def select(self) -> int:  # strata idx
        """
        Returns a strata index
        -------
        """
        if not self.is_select_call:
            raise ValueError("next call should be `next`")
        result = self._select()
        self.is_select_call = False
        return result

    def set(self, idx: int, label: bool):
        """
            set label for the pair selected
            params:
                idx: index of the pair in the array
                label: 0 or 1
        """
        if self.is_select_call:
            raise ValueError("next call should be `select`")
        self._set(idx, label)
        self.is_select_call = True
        self.pairs_sampled.append((idx, label))

    def f_score_history(self) -> List[float]:
        """
        Returns the f score estimated at each iteration.
        -------
        """
        return self.f_scores

    def pairs(self) -> List[int]:
        """
        Returns the pairs sampled at each iterations in order
        -------
        """
        return self.pairs_sampled



