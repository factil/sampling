# Efficient evaluation of classifiers
this repo includes several efficient methods to estimate the f score of
a given classifier through labelling
1. StratifiedUniformSampler
2. DruckSampler
3. OasisSampler
### Installation
```
git clone https://github.com/factil/sampling.git && cd sampling
python setup.py install
```

### Dev Quickstart
```
pip install pipenv
# from Pipfile
pipenv install --dev

# from Pipfile.lock
pipenv install --ignore-pipfile
```

### Client code
```python
import json
import numpy as np
from functools import partial
from sampling.utility import scores2probs
from sampling.sampler import Sampler
from sampling.oasis import OASISSampler


data = json.load(open('data.json'))
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
```