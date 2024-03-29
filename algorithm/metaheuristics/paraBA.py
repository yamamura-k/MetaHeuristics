"""
Sample implementation of Artificial Bee Colony Algorithm.

Reference : https://link.springer.com/content/pdf/10.1007/s10898-007-9149-x.pdf
"""

import ctypes
from multiprocessing import Pool

import numpy as np
from utils import numpy_to_value, setup_logger, value_to_numpy
from utils.common import ResultManager

np.random.seed(0)
logger = setup_logger(__name__)


def step(i, t):
    f[i] = f_min + (f_max - f_min)*np.random.uniform(0, 1)
    v[i*dimension:(i+1)*dimension] += (x[i*dimension:(i+1)
                                         * dimension] - best_x) * f[i]
    x_t = x[i*dimension:(i+1)*dimension] + v[i*dimension:(i+1)*dimension]
    obj_new = None
    obj_t = objective(x_t)

    if np.random.rand() > r[i]:
        eps = np.random.uniform(-1, 1)
        idx = np.random.randint(0, selection_max)
        x_new = x[idx*dimension:(idx+1)*dimension].copy()
        x_new += eps * np.average(A)
        obj_new = objective(x_new)

    x_random = np.random.uniform(*objective.boundaries, size=dimension)
    obj_random = objective(x_random)
    if (obj_new is None or obj_t > obj_new) and obj_t > obj_random:
        if obj_new is None or obj_new >= obj_random:
            value_to_numpy(x_share)[i*dimension:(i+1)*dimension] = x_random
            if obj_random < obj_bests[i]:
                obj_bests[i] = obj_random
        else:
            value_to_numpy(x_share)[i*dimension:(i+1)*dimension] = x_new
            if obj_new < obj_bests[i]:
                obj_bests[i] = obj_new
    else:
        value_to_numpy(x_share)[i*dimension:(i+1)*dimension] = x_t
        if obj_t < obj_bests[i]:
            obj_bests[i] = obj_t

    r[i] = r0[i] * (1-np.exp(-gamma*t))
    A[i] *= alpha


def init(_x, _x_share, _v, _r, _r0, _f, _A, _f_min, _f_max, _sel_max, _alpha, _gamma, _obj_best, _objective, _best_x):
    global x
    global x_share
    global v
    global r
    global r0
    global f
    global f_min
    global f_max
    global selection_max
    global alpha
    global gamma
    global A
    global dimension
    global obj_bests
    global objective
    global best_x
    dimension = _x.shape[0] // _r.shape[0]
    x = _x
    x_share = _x_share
    v = _v
    r = _r
    r0 = _r0
    f = _f
    A = _A
    f_min = _f_min
    f_max = _f_max
    selection_max = _sel_max
    alpha = _alpha
    gamma = _gamma
    obj_bests = _obj_best
    objective = _objective
    best_x = _best_x


def minimize(dimension, objective, max_iter, num_population=100, f_min=0,
             f_max=100, selection_max=10, alpha=0.9, gamma=0.9, num_cpu=None, *args, **kwargs):
    x = np.random.uniform(*objective.boundaries, size=num_population*dimension)
    x_share = numpy_to_value(x, ctypes.c_double)
    v = numpy_to_value(np.random.random(
        num_population*dimension), ctypes.c_double)
    f = np.random.uniform(f_min, f_max, size=num_population)
    A = np.random.uniform(1, 2, size=num_population)
    r = np.random.uniform(0, 1, size=num_population)
    r0 = r.copy()

    obj_best = np.inf
    best_x = None
    for i in range(num_population):
        obj_tmp = objective(x[i*dimension:(i+1)*dimension])
        if obj_tmp < obj_best:
            obj_best = obj_tmp
            best_x = x[i*dimension:(i+1)*dimension].copy()
    obj_bests = numpy_to_value(
        np.array([obj_best]*num_population), ctypes.c_double)
    result = ResultManager(objective, __name__, logger, *args, **kwargs)
    result.post_process_per_iter(
        x.reshape(num_population, dimension), best_x.reshape((dimension, 1)), -1)
    with Pool(num_cpu, initializer=init, initargs=(x, x_share, v, r, r0, f, A, f_min, f_max, selection_max, alpha, gamma, obj_bests, objective, best_x)) as p:
        for t in range(max_iter):
            p.starmap(step, [(i, t)
                      for i in range(num_population)])

            x = value_to_numpy(x_share)

            for i in range(num_population):
                if obj_best > obj_bests[i]:
                    best_x = x[i*dimension:(i+1)*dimension].copy()
                    obj_best = obj_bests[i]
                else:
                    obj_bests[i] = obj_best

            x = list(x.reshape(num_population, dimension))
            x.sort(key=lambda s: objective(s))
            x = np.array(x)
            if result.post_process_per_iter(x, best_x.reshape((dimension, 1)), t):
                break
            x = x.reshape(num_population*dimension)

    return result
