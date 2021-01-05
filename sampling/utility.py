import numpy as np
from scipy.special import expit


def scores2probs(scores, eps=0.01):
    max_extreme_score = np.max(np.abs(scores))
    k = np.log((1 - eps) / eps) / max_extreme_score  # scale factor
    return expit(k * scores)


def compute_f_score(alpha, true_positves, false_positives, false_negatives):
    """Calculate the weighted F-measure"""
    num = true_positves
    den = np.float64(alpha * (true_positves + false_positives) +\
                     (1 - alpha) * (true_positves + false_negatives))
    with np.errstate(divide='ignore', invalid='ignore'):
        return num / den


def equal_length(x, y):
    return len(x) == len(y)


def is_between_zero_and_one(x):
    return np.all((0 <= x) & (x <= 1))