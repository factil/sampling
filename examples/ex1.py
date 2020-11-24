import json
from functools import partial
import numpy as np
from utility import scores2probs
from passive import PassiveSampler
from sawade import ImportanceSampler
from druck import DruckSampler


def oracle(labels, idx):
    return labels[idx]


if __name__ == '__main__':
    # from neil's example dataset
    data = json.load(open('data.json'))
    labels = np.array(data['labels'])
    scores = np.array(data['scores'])
    preds = np.array(data['preds'])
    probs = scores2probs(scores)

    # initialize all samplers
    oracle = partial(oracle, labels)
    passive_sampler = PassiveSampler(0.5, preds, probs, oracle)
    importance_sampler = ImportanceSampler(0.5, preds, probs, oracle)
    druck_sampler = DruckSampler(0.5, preds, probs, oracle)

    # sample
    passive_sampler.sample_distinct(5000)
    importance_sampler.sample_distinct(5000)
    druck_sampler.sample_distinct(5000)

    # its likely that f score from passive sampler is all NAN, consider increase number of samples

    # f score estimates
    print(f"{passive_sampler.f_score_history()=}")
    print(f"{importance_sampler.f_score_history()=}")
    print(f"{druck_sampler.f_score_history()=}")
