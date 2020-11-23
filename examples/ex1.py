import json
from functools import partial
import numpy as np
from utility import scores2probs
from passive import PassiveSampler
from sawade import ImportanceSampler


def oracle(labels, idx):
    return labels[idx]


if __name__ == '__main__':
    data = json.load(open('data.json'))
    labels = np.array(data['labels'])
    scores = np.array(data['scores'])
    preds = np.array(data['preds'])
    probs = scores2probs(scores)
    oracle = partial(oracle, labels)
    passive_sampler = PassiveSampler(0.5, preds, probs, oracle)
    importance_sampler = ImportanceSampler(0.5, preds, probs, oracle)
    passive_sampler.sample_distinct(5000)
    importance_sampler.sample_distinct(5000)
    # its likely that f score from passive sampler is all NAN, consider increase number of samples
    print(passive_sampler.f_score_history())
    print(importance_sampler.f_score_history())
