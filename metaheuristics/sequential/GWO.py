import numpy as np

from .utils import randomize


def optimize(dimension, num_population, objective, max_iter, *args, **kwargs):
    x = randomize((num_population, dimension), objective)
