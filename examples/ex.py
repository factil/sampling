import json
import numpy as np
from functools import partial
from sampling.utility import scores2probs
from sampling.sampler import Sampler
from sampling.uniform_stratified import StratifiedUniformSampler


data = json.load(open('data.json'))
labels = np.array(data['labels'])
scores = np.array(data['scores'])
preds = np.array(data['preds'])
probs = scores2probs(scores)


def oracle(labels, idx):
    return labels[idx]


# initialize all samplers
oracle = partial(oracle, labels)
sampler = Sampler(StratifiedUniformSampler, 0.5, probs, [], [], n_strata=10)
sampler.sample(oracle, 5000)
print(sampler.f_score_history())