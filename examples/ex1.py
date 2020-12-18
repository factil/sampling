import sys

sys.path.append('.')

import json
from functools import partial
import numpy as np
from utility import scores2probs
from passive import PassiveSampler
from sawade import ImportanceSampler
from druck import DruckSampler
from _oasis import OASISSampler


def oracle(labels, idx):
    return labels[idx]


if __name__ == '__main__':
    # from neil's example dataset
    data = json.load(open('examples/data.json'))
    labels = np.array(data['labels'])
    scores = np.array(data['scores'])
    preds = np.array(data['preds'])
    probs = scores2probs(scores)

    # initialize all samplers
    oracle = partial(oracle, labels)
    for cls in [PassiveSampler, ImportanceSampler, DruckSampler, OASISSampler]:
        sampler = cls(0.5, preds, probs, oracle)
        sampler.sample_distinct(5000)
        print(f"{cls.__name__} f score history:")
        print(sampler.f_score_history())
