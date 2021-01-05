from typing import List, Optional, Tuple
from .stratification import Strata


class LabelledPairs:
    def __init__(self, strata: Strata, pair_ids: List, scores: List, labels: List):
        self.strata = strata
        self.labelled_pairs = [[] for _ in range(len(self.strata))]
        for i, s, l in zip(pair_ids, scores, labels):
            self.labelled_pairs[self.strata.stratum_idx_for_score(s)].append((i, l))

    def sample_at_stratum(self, x) -> Optional[Tuple]: # pair idx
        stratum = self.labelled_pairs[x]
        if stratum:
            return stratum.pop()
