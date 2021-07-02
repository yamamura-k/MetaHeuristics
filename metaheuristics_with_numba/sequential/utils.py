import numpy as np
from numba import njit


@njit
def randomize(shape, objective):
    try:
        return np.random.uniform(*objective.boundaries, size=shape)
    except:
        return np.random.random(size=shape)
