import numpy as np


def randomize(shape, objective):
    try:
        return np.random.uniform(*objective.boundaries, size=shape)
    except:
        return np.random.random(size=shape)
