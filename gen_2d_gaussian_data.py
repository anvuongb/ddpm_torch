import numpy as np
import pickle

class_mean = [0, -10, 10]
class_var = [1, 2, 3]
class_probs = [0.4, 0.35, 0.25]

if np.sum(class_probs) != 1.0:
    raise ("Total class probs should be 1.0")

N = 1000000
s = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], N)
begin = 0
for idx, cl in enumerate(class_probs):
    end = begin + int(N * cl)
    s[begin:end] = class_mean[idx] + np.sqrt(class_var[idx]) * s[begin:end]
    begin = end
xs, ys = s[:, 0], s[:, 1]
with open("./data/2d_gaussians/train.pickle", "wb") as f:
    pickle.dump(s, f)