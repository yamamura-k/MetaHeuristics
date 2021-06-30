"""
Sample implementation of Artificial Bee Colony Algorithm.

Reference : https://link.springer.com/content/pdf/10.1007/s10898-007-9149-x.pdf
"""

import ctypes
import os
from multiprocessing import Pool

import numpy as np

from benchmarks import (ackley, different_power, griewank, k_tablet,
                        rosenbrock, sphere, styblinski, weighted_sphere)
from utils import numpy_to_value, value_to_numpy


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
        if obj_new is None or obj_new <= obj_random:
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


def optimize(dimension, num_population, objective, max_iter, f_min=0,
             f_max=100, selection_max=10, alpha=0.9, gamma=0.9, num_cpu=None):
    x = np.random.uniform(*objective.boundaries, size=num_population*dimension)
    x_share = numpy_to_value(x, ctypes.c_double)
    v = numpy_to_value(np.random.random(
        num_population*dimension), ctypes.c_double)
    f = np.random.uniform(f_min, f_max, size=num_population)
    A = np.random.uniform(1, 2, size=num_population)
    r = np.random.uniform(0, 1, size=num_population)
    r0 = r.copy()

    obj_best = float('inf')
    best_x = None
    for i in range(num_population):
        obj_tmp = objective(x[i*dimension:(i+1)*dimension])
        if obj_tmp < obj_best:
            obj_best = obj_tmp
            best_x = x[i*dimension:(i+1)*dimension].copy()
    obj_bests = numpy_to_value(
        np.array([obj_best]*num_population), ctypes.c_double)
    pos1 = []
    pos2 = []
    best_pos1 = []
    best_pos2 = []
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
            x = np.array(x).reshape(num_population*dimension)

            pos1.append([x[i*dimension:(i+1)*dimension][0]
                        for i in range(num_population)])
            pos2.append([x[i*dimension:(i+1)*dimension][1]
                        for i in range(num_population)])
            best_pos1.append(best_x[0])
            best_pos2.append(best_x[1])

    return best_x, obj_best, (pos1, pos2, best_pos1, best_pos2)


def main():
    bench_funcs = [
        ackley(), sphere(), rosenbrock(), styblinski(), k_tablet(),
        weighted_sphere(), different_power(), griewank()]

    dimension = 2
    num_population = 15
    max_iter = 30
    for f in bench_funcs:
        print(f.name, "minimum =", f.opt)
        position, best, logs = optimize(
            dimension, num_population, f, max_iter)
        print("minimum is", best)
        print("position is", *position)
        f.plot(*logs, algo_name="paraBA")


if __name__ == '__main__':
    main()
