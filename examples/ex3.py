import oasis

data = oasis.Data()
data.read_h5('Amazon-GoogleProducts-test.h5')
def oracle(idx):
    return data.labels[idx]

alpha = 0.5      #: corresponds to F1-score
n_labels = 5000  #: stop sampling after querying this number of labels
max_iter = 1e6   #: maximu


f_scores = []
for i in range(50):
    smplr = oasis.OASISSampler(alpha, data.preds, data.scores, oracle, max_iter=max_iter)
    smplr.sample_distinct(n_labels)
    f_scores.append(smplr.estimate_[-1])

print(f_scores)
print(max(f_scores))
print(min(f_scores))