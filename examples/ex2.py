import sys

sys.path.append('.')

import json
from functools import partial
import numpy as np
import tqdm
from oasis import OASISSampler
from mine import MySampler
from utility import scores2probs

OASISSampler.f_score_history = lambda x: x.estimate_


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
    n_runs = 30
    f_scores = np.empty(n_runs)
    for sampler_gen in [lambda: OASISSampler(0.5, preds, scores, oracle), lambda:MySampler(0.5, preds, probs, oracle)]:
        print(sampler_gen().__class__.__name__)
        for i in tqdm.tqdm(range(n_runs)):
            sampler = sampler_gen()
            sampler.sample_distinct(5000)
            f_scores[i] = sampler.f_score_history()[-1]
        print("estimated f score")
        print(f"{list(f_scores)=}")
        print(f"{np.mean(f_scores)=}")
        print(f"{np.std(f_scores)=}")
        print(f"{f_scores.max()=}")
        print(f"{f_scores.min()=}")
        print(f"{'*' + '-' * 120 + '*'}")
