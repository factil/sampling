from typing import Callable
from abc import ABC, abstractmethod
import numpy as np

from utility import is_between_zero_and_one


class BaseSampler(ABC):
    def __init__(self,
                 alpha: float,
                 predictions,
                 scores,
                 oracle: Callable,
                 max_iter=None):

        if not is_between_zero_and_one(alpha):
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.oracle = oracle
        if not is_between_zero_and_one(scores):
            raise ValueError("scores must be between 0 and 1")
        self.scores = scores

        self.predictions = predictions
        self.n_items = len(self.predictions)
        self.max_iter = self.n_items if (max_iter is None) else int(max_iter)
        self.idx = 0

        self.queried_oracle = np.repeat(False, self.max_iter)
        self.cached_labels = np.repeat(np.nan, self.n_items)
        self.f_scores = np.repeat(np.nan, self.max_iter)

    @abstractmethod
    def select_single_item(self, sample_with_replacement: bool, **kwargs):
        pass

    @abstractmethod
    def update_estimate_and_sampler(self, ell, ell_hat, weight, **kwargs):
        pass

    def f_score_history(self):
        return self.f_scores[:self.idx]

    def query_label(self, loc):
        """Query the label for the item with index `loc`. Preferentially
        queries the label from the cache, but if not yet cached, queries the
        oracle.

        Returns
        -------
        int
            the true label "0" or "1".
        """
        # Try to get label from cache

        ell = self.cached_labels[loc]
        if not np.isnan(ell):
            return ell

            # Label has not been cached. Need to query oracle
        ell = self.oracle(loc)
        if ell not in [0, 1]:
            raise Exception("Oracle provided an invalid label.")
        # TODO Gracefully handle errors from oracle?
        self.queried_oracle[self.idx] = True
        self.cached_labels[loc] = ell

        return ell

    def sample(self, n_to_sample: int, sample_with_replacement=True, **kwargs):
        """Sample a sequence of items from the pool (with replacement)

        Parameters
        ----------
        n_to_sample : positive int
            number of items to sample
        sample_with_replacement : true
        """
        for _ in range(n_to_sample):
            loc, weight, extra_info = self.select_single_item(sample_with_replacement, **kwargs)
            # Query label
            ell = self.query_label(loc)
            # Get predictions
            ell_hat = self.predictions[loc]
            if isinstance(extra_info, dict):
                self.update_estimate_and_sampler(ell, ell_hat, weight, **extra_info)
            else:
                self.update_estimate_and_sampler(ell, ell_hat, weight)
            self.idx += 1

    def sample_distinct(self, n_to_sample):
        self.sample(n_to_sample, sample_with_replacement=False)
