import numpy as np
from stratification import Strata, stratify_by_cum_sqrt_f_method
from sampler_abc import BaseSampler
from utility import compute_f_score


class DruckSampler(BaseSampler):
    """Stratified sampling for estimation of the weighted F-measure

    Estimates the quantity::

            TP / (alpha * (TP + FP) + (1 - alpha) * (TP + FN))

    on a finite pool by sampling items according to an adaptive instrumental
    distribution that minimises asymptotic variance. See reference
    [Druck2011]_ for details.

    Parameters
    ----------
    alpha : float
        Weight for the F-measure. Valid weights are on the interval [0, 1].
        ``alpha == 1`` corresponds to precision, ``alpha == 0`` corresponds to
        recall, and ``alpha == 0.5`` corresponds to the balanced F-measure.

    predictions : array-like, shape=(n_items,n_class)
        Predicted labels for the items in the pool. Rows represent items and
        columns represent different classifiers under evaluation (i.e. more
        than one classifier may be evaluated in parallel). Valid labels are 0
        or 1.

    scores : array-like, shape=(n_items,)
        Scores which quantify the confidence in the classifiers' predictions.
        Rows represent items and columns represent different classifiers under
        evaluation. High scores indicate a high confidence that the true label
        is 1 (and vice versa for label 0). It is recommended that the scores
        be scaled to the interval [0,1]. If the scores lie outside [0,1] they
        will be automatically re-scaled by applying the logisitic function.

    oracle : function
        Function that returns ground truth labels for items in the pool. The
        function should take an item identifier as input (i.e. its
        corresponding row index) and return the ground truth label. Valid
        labels are 0 or 1.

    proba : array-like, dtype=bool, shape=(n_class,), optional, default None
        Indicates whether the scores are probabilistic, i.e. on the interval
        [0, 1] for each classifier under evaluation. If proba is False for
        a classifier, then the corresponding scores will be re-scaled by
        applying the logistic function. If None, proba will default to
        False for all classifiers.

    strata : Strata instance, optional, default None
        Describes how to stratify the pool. If not given, the stratification
        will be done automatically based on the scores given. Additional
        keyword arguments may be passed to control this automatic
        stratification (see below).

    max_iter : int, optional, default None
        Maximum number of iterations to expect for pre-allocating arrays.
        Once this limit is reached, sampling can no longer continue. If no
        value is given, defaults to n_items.

    replace : boolean, optional, default True
        Whether to sample with or without replacement.

    Other Parameters
    ----------------
    opt_class : array-like, dtype=bool, shape=(n_class,), optional, default None
        Indicates which classifier scores to use when stratifying the pool (if
        `strata` is None). If opt_class is False for a classifier, then its
        scores will not be used in calculating the strata, however estimates of
        its performance will still be calculated.

    identifiers : array-like, optional, default None
        Unique identifiers for the items in the pool. Must match the row order
        of the "predictions" parameter. If no value is given, defaults to
        [0, 1, ..., n_items].

    debug : bool, optional, default False
        Whether to print out verbose debugging information.

    Attributes
    ----------
    estimate_ : numpy.ndarray
        F-measure estimates for each iteration.

    queried_oracle_ : numpy.ndarray
        Records whether the oracle was queried at each iteration (True) or
        whether a cached label was used (False).

    cached_labels_ : numpy.ndarray, shape=(n_items,)
        Previously sampled ground truth labels for the items in the pool. Items
        which have not had their labels queried are recorded as NaNs. The order
        of the items matches the row order for the "predictions" parameter.

    t_ : int
        Iteration index.

    References
    ----------
    .. [Druck2011] G. Druck and A. McCallum, "Toward Interactive Training and
       Evaluation," in Proceedings of the 20th ACM International Conference on
       Information and Knowledge Management, 2011, pp. 947–956.

    """
    def __init__(self, alpha, predictions, scores, oracle, max_iter=None):
        super(DruckSampler, self).__init__(alpha, predictions, scores, oracle, max_iter=max_iter)

        self.strata = Strata(stratify_by_cum_sqrt_f_method(scores))

        #: Number of TP, PP, P sampled per stratum
        self.tp_strata = np.zeros(self.strata.n_strata)
        self.pp_strata = np.zeros(self.strata.n_strata)
        self.pp_st = np.zeros(self.strata.n_strata)

        # fill each of strata with 2 items as it is described in the paper
        for i in range(2):
            for j in range(self.strata.n_strata):
                self.sample(1, stratum_idx=j)

    def select_next_item(self, sample_with_replacement: bool, **kwargs):
        stratum_idx = kwargs.get('stratum_idx')
        if stratum_idx is not None:
            #: Sample in given stratum
            loc = self.strata.sample_in_stratum(stratum_idx, replace=sample_with_replacement)
        else:
            loc, stratum_idx = self.strata.sample(replace=sample_with_replacement)

        return loc, 1, {'stratum_idx': stratum_idx}

    def update_estimate_and_sampler(self, ell, ell_hat, weight, **kwargs):
        """Update the estimate after querying the label for an item"""

        stratum_idx = kwargs['stratum_idx']
        self.tp_strata[stratum_idx] += ell_hat * ell * weight
        self.pp_strata[stratum_idx] += ell_hat * weight
        self.pp_st[stratum_idx] += ell * weight

        p_rates = self.pp_st / self.strata.n_sampled_in_each_strata
        tp_rates = self.tp_strata / self.strata.n_sampled_in_each_strata
        pp_rates = self.pp_strata / self.strata.n_sampled_in_each_strata

        sizes = self.strata.sizes
        #: Estimate number of TP, PP, P
        tp = np.inner(tp_rates, sizes)
        pp = np.inner(pp_rates, sizes)
        p = np.inner(p_rates, sizes)
        fn = tp - pp
        fp = p - pp
        f_score = compute_f_score(self.alpha, tp, fp, fn)

        #: Update model estimate (with prior)
        self.f_scores[self.idx] = f_score
