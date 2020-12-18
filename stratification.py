import random
import bisect
import numpy as np
from collections import Counter


def frequency_counts(arr, n):
    result = np.zeros(n)
    for i in arr:
        result[i] += 1
    return result


def bin_width_using_freedman_diaconis_rule(obs):
    IQR = np.percentile(obs, 75) - np.percentile(obs, 25)
    N = len(obs)
    return 2 * IQR * N ** (-1 / 3)


def uniform_strata_method(scores, n_strata):
    bounds = [i/n_strata for i in range(n_strata+1)]
    allocations = [bisect.bisect(bounds, x)-1 for x in scores]
    print(Counter(allocations))
    return allocations, bounds


def stratify_by_equal_size_method(scores):
    strata_width = bin_width_using_freedman_diaconis_rule(scores)
    goal_n_strata = np.ceil(np.ptp(scores) / strata_width).astype(np.int)
    n_items = len(scores)
    sorted_ids = scores.argsort()
    quotient = n_items // goal_n_strata
    remainder = n_items % goal_n_strata
    allocations = np.empty(n_items, dtype='int')
    st_pops = (np.repeat(quotient, goal_n_strata) + np.concatenate(
        (np.ones(remainder), np.zeros(goal_n_strata - remainder)))) \
        .cumsum().astype(int)

    bounds = [scores.min()]
    for k, (start, end) in enumerate(zip(np.concatenate((np.zeros(1), st_pops)).astype(int), st_pops)):
        allocations[sorted_ids[start:end]] = k
        bounds.append(scores[sorted_ids[end - 1]])

    return allocations, bounds


def stratify_by_cum_sqrt_f_method(scores):
    score_width = bin_width_using_freedman_diaconis_rule(scores)
    n_bins = np.ceil(np.ptp(scores) / score_width).astype(int)
    counts, score_bins = np.histogram(scores, bins=n_bins)
    csf = np.sqrt(counts).cumsum() # cum sqrt(F)
    strata_width = bin_width_using_freedman_diaconis_rule(csf)
    bounds = []
    j = 0
    for x, sb in zip(csf, score_bins):
        if x >= strata_width * j:
            bounds.append(sb)
            j += 1

    bounds.append(score_bins[-1])
    # add margin
    bounds[0] = max(bounds[0]-0.01, 0)
    bounds[-1] = min(bounds[-1]+0.01, 1)

    return np.digitize(scores, bins=bounds, right=True) - 1, bounds


class Strata:
    def __init__(self, allocations, bounds):
        self.n_strata = np.max(allocations) + 1
        self.bounds = bounds
        self.strata = [[] for _ in range(self.n_strata)]

        for i, si in enumerate(allocations):
            self.strata[si].append(i)

        self.sizes = np.array([len(x) for x in self.strata])

    def __len__(self):
        return len(self.bounds) - 1

    def sample_in_strata(self, i):
        return random.choice(self.strata[i])

    def stratum_idx_for_score(self, x):
        return bisect.bisect(self.bounds, x) - 1

    def intra_mean(self, a):
        return np.array([np.mean(a[x]) for x in self.strata])

    def strata_bounds(self):
        return self.bounds[:]

    @classmethod
    def from_esm(cls, scores):
        """
        stratify with equal size method
        """
        allocations, bounds = stratify_by_cum_sqrt_f_method(scores)
        return Strata(allocations, bounds)

    @classmethod
    def from_usm(cls, scores, n=10):
        allocations, bounds = uniform_strata_method(scores, n)
        return Strata(allocations, bounds)

    @classmethod
    def from_csf(cls, scores):
        """
        stratify with cum sqrt f method
        """
        allocations, bounds = stratify_by_cum_sqrt_f_method(scores)
        return Strata(allocations, bounds)