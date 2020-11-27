import json
from functools import partial
import numpy as np
import tqdm
from oasis import OASISSampler

from utility import scores2probs



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
    n_runs = 50
    f_scores = np.empty(n_runs)
    for i in tqdm.tqdm(range(n_runs)):
        sampler = OASISSampler(0.5, preds, probs, oracle)
        sampler.sample_distinct(5000)
        f_scores[i] = sampler.estimate_[-1]
    print("estimated f score")
    print(f"{np.mean(f_scores)=}")
    print(f"{np.std(f_scores)=}")
