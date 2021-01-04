import json
import numpy as np
from functools import partial
from utility import scores2probs
from sampler import Sampler
from _oasis import OASISSampler




data = json.load(open('examples/data.json'))
labels = np.array(data['labels'])
scores = np.array(data['scores'])
preds = np.array(data['preds'])
probs = scores2probs(scores)


def oracle(labels, idx):
    return labels[idx]


# initialize all samplers
oracle = partial(oracle, labels)
sampler = Sampler(OASISSampler, 0.5, probs, [], [])
sampler.sample(oracle, 5000)
print(sampler.f_score_history())