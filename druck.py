import numpy as np
from sampler_abc import SamplerInternal, Strata
from utility import compute_f_score


class DruckSampler(SamplerInternal):

    def __init__(self, alpha, scores, strata: Strata, threshold=0.5, **kwargs):
        super().__init__(alpha, scores, threshold=threshold, **kwargs)
        self.strata = strata
        self.n_strata = len(self.strata)

        #: Number of TP, PP, P sampled per stratum
        self.tp_strata = np.zeros(self.n_strata)
        self.pp_strata = np.zeros(self.n_strata)
        self.pp_st = np.zeros(self.n_strata)

        # fill each of strata with 2 items as it is described in the paper
        self.pre_sampling = [x for x in range(self.n_strata) for _ in range(2)]
        self.number_sampled_at_each_strata = np.zeros(self.n_strata)
        self.pmf = self.strata.sizes / len(self.allocations)
        self.current_strata = None

    def _select(self) -> int:
        idx = None
        if self.pre_sampling:
            idx = self.pre_sampling.pop()

        if not idx:
            idx = np.random.choice(np.arange(self.n_strata), p=self.pmf)

        self.current_strata = idx
        self.number_sampled_at_each_strata[idx] += 1
        return idx

    def _set(self, idx: int,  label: bool):
        ell = label
        ell_hat = self.scores[idx]
        self.tp_strata[self.current_strata] += ell_hat * ell
        self.pp_strata[self.current_strata] += ell_hat
        self.pp_st[self.current_strata] += ell

        p_rates = self.pp_st / self.number_sampled_at_each_strata
        tp_rates = self.tp_strata / self.number_sampled_at_each_strata
        pp_rates = self.pp_strata / self.number_sampled_at_each_strata

        sizes = self.strata.sizes
        #: Estimate number of TP, PP, P
        tp = np.inner(tp_rates, sizes)
        pp = np.inner(pp_rates, sizes)
        p = np.inner(p_rates, sizes)
        fn = tp - pp
        fp = p - pp
        f_score = compute_f_score(self.alpha, tp, fp, fn)

        #: Update model estimate (with prior)
        self.f_scores.append(f_score)
