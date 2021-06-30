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
from utils import numpy_to_value, value_to_numpy, value_to_numpy2


def step(i, t, x, best_x, obj_best):
    f[i] = f_min + (f_max - f_min)*np.random.uniform(0, 1)
    v[i] += (x[i] - best_x) * f[i]
    x_t = x[i] + v[i]
    obj_new = None
    obj_t = objective(x_t)

    if np.random.rand() > r[i]:
        eps = np.random.uniform(-1, 1)
        idx = np.random.randint(0, selection_max)
        x_new = x[idx].copy()
        x_new += eps * np.average(A)
        obj_new = objective(x_new)

    x_random = np.random.random(dimension)
    obj_random = objective(x_random)
    if (obj_new is None or obj_t > obj_new) and obj_t > obj_random:
        if obj_new is None or obj_new <= obj_random:
            x[i] = x_random
            if obj_random < obj_best:
                obj_best = obj_random
                best_x = x[i].copy()
        else:
            x[i] = x_new
            if obj_new < obj_best:
                obj_best = obj_new
                best_x = x[i].copy()
    else:
        x[i] = x_t
        if obj_t < obj_best:
            obj_best = obj_t
            best_x = x[i].copy()

    r[i] = r0[i] * (1-np.exp(-gamma*t))
    A[i] *= alpha
    x = list(x)
    x.sort(key=lambda s: objective(s))
    x = np.array(x)


def init(_x, _v, _r, _r0, _f, _A, _f_min, _f_max, _sel_max, _alpha, _gamma, _obj_best, _objective, _best_x):
    global x
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
    global obj_best
    global objective
    global best_x
    _, dimension = _x.shape
    x = _x
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
    obj_best = _obj_best
    objective = _objective
    best_x = _best_x


def optimize(dimension, num_population, objective, max_iter, f_min=0,
             f_max=100, selection_max=10, alpha=0.9, gamma=0.9, num_cpu=None):
    x = np.random.random((num_population, dimension))
    v = np.random.random((num_population, dimension))
    f = np.random.uniform(f_min, f_max, size=num_population)
    A = np.random.uniform(1, 2, size=num_population)
    r = np.random.uniform(0, 1, size=num_population)
    r0 = r.copy()
    L = list(range(num_population))

    obj_best = float('inf')
    best_x = None
    for i in range(num_population):
        obj_tmp = objective(x[i])
        if obj_tmp < obj_best:
            obj_best = obj_tmp
            best_x = x[i].copy()

    pos1 = []
    pos2 = []
    best_pos1 = []
    best_pos2 = []
    with Pool(num_cpu, initializer=init, initargs=(x, v, r, r0, f, A, f_min, f_max, selection_max, alpha, gamma, obj_best, objective, best_x)) as p:
        for t in range(max_iter):
            p.starmap(step, [(i, t, x, best_x, obj_best)
                      for i in range(num_population)])

            pos1.append([x[0] for x in x])
            pos2.append([x[1] for x in x])
            best_pos1.append(best_x[0])
            best_pos2.append(best_x[1])

    return best_x, obj_best, (pos1, pos2, best_pos1, best_pos2)


def main():
    bench_funcs = [
        ackley(), sphere(), rosenbrock(), styblinski(), k_tablet(),
        weighted_sphere(), different_power(), griewank()]

    dimension = 2
    num_population = 50
    max_iter = 100
    for f in bench_funcs:
        print(f.name, "minimum =", f.opt)
        position, best, logs = optimize(
            dimension, num_population, f, max_iter)
        print("minimum is", best)
        print("position is", *position)
        f.plot(*logs, algo_name="paraBA")


if __name__ == '__main__':
    main()
