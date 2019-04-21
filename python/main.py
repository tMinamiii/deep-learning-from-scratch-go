import numpy as np


def relu(x):
    if x > 0:
        return x
    else:
        return 0


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
