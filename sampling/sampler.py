from .stratification import Strata
from .utility import equal_length
from .labelled_pairs import LabelledPairs


class Sampler:
    def __init__(self,
                 internal_cls,
                 alpha,
                 scores,
                 labelled_pair_ids,
                 labels,
                 threshold=0.5, **kwargs):

        if n_strata := kwargs.get('n_strata'):
            self.strata = Strata.from_usm(scores, n_strata)
        else:
            self.strata = Strata.from_csf(scores)

        if not equal_length(labelled_pair_ids, labels):
            raise ValueError("labelled_pair_ids must match labels")

        self.internal = internal_cls(alpha,
                                     scores,
                                     self.strata,
                                     threshold=threshold, **kwargs)
        self.labelled_pairs = LabelledPairs(self.strata, labelled_pair_ids, scores[labelled_pair_ids], labels)

    def sample(self, oracle_func, n: int):
        i = 0
        while i < n:
            stratum_idx = self.internal.select()
            # find a labelled pair if available
            result = self.labelled_pairs.sample_at_stratum(stratum_idx)
            if result is not None:
                pair_id, label = result

            # sample an unknown pair for labelling
            else:
                pair_id = self.strata.sample_in_strata(stratum_idx)
                label = oracle_func(pair_id)

            # set the result
            self.internal.set(pair_id, label)
            i += 1

    def f_score_history(self):
        return self.internal.f_score_history()
