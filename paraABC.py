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


def update(i):
    x_i = x[i*dimension:(i+1)*dimension].copy()
    cnt_share[i] += 1
    j = np.random.randint(0, dimension-1)
    k = np.random.randint(0, num_population-1)
    phi = np.random.normal()
    x_i[j] -= phi*(x_i[j] - x[k*dimension + j])
    v_new = objective(x_i)
    if v_new <= v_share[i]:
        value_to_numpy(x_share)[
            i*dimension:(i+1)*dimension] = x_i
        v_share[i] = v_new


def random_update(i):
    x_i = np.random.uniform(*f.boundaries, size=dimension)
    v_new = objective(x_i)
    if v_new <= v_share[i]:
        value_to_numpy(x_share)[
            i*dimension:(i+1)*dimension] = x_i
        v_share[i] = v_new
        cnt_share[i] = 1


def init(_x, _v, _cnt, x_share_, v_share_, cnt_share_, _objective):
    global x_share
    global v_share
    global cnt_share
    global cnt
    global x
    global v
    global objective
    global dimension
    global num_population
    product = _x.shape[0]
    num_population = _v.shape[0]
    dimension = product // num_population
    x = _x
    v = _v
    objective = _objective
    cnt = _cnt
    x_share = x_share_
    v_share = v_share_
    cnt_share = cnt_share_


def optimize(dimension, num_population, max_visit, objective, C, num_cpu=None):

    if num_cpu is None:
        num_cpu = os.cpu_count()
    best_obj = float('inf')
    best_x = None
    # step1 : initialization
    x = np.random.uniform(*objective.boundaries, size=num_population*dimension)
    v = np.array([objective(x[i*dimension:(i+1)*dimension])
                 for i in range(num_population)])
    all_candidates = list(range(num_population))
    cnt = np.zeros(num_population)
    x_share = numpy_to_value(x, ctypes.c_double)
    v_share = numpy_to_value(v, ctypes.c_double)
    cnt_share = numpy_to_value(cnt, ctypes.c_int, 0)

    pos1 = []
    pos2 = []
    best_pos1 = []
    best_pos2 = []

    with Pool(num_cpu, initializer=init, initargs=(x, v, cnt, x_share, v_share, cnt_share, objective)) as p:
        for c in range(1, C+1):
            # employed bees
            # result = p.map_async(update, all_candidates)
            p.map(update, all_candidates)

            # onlooker bees
            x = value_to_numpy(x_share)
            v = value_to_numpy(v_share)
            # calculate selection probabirities
            if (v >= 0).all():
                probs = v / np.sum(v)
            else:
                m = np.min(v)
                w = v - m
                probs = w / np.sum(w)
            probs = 1 - probs
            probs /= np.sum(probs)
            onlooker_list = np.random.choice(
                all_candidates, size=num_population, p=probs)
            p.map(update, onlooker_list.tolist())

            # scouts
            x = value_to_numpy(x_share)
            v = value_to_numpy(v_share)
            cnt = value_to_numpy2(cnt_share)
            scout_list = np.where(cnt == max_visit)[0]
            p.map(random_update, scout_list)

            x = value_to_numpy(x_share)
            v = value_to_numpy(v_share)

            m = np.min(v)
            pos = [x[i*dimension:(i+1)*dimension]
                   for i in range(num_population)]
            pos1.append([t[0] for t in pos])
            pos2.append([t[1] for t in pos])
            best_pos = np.where(v == m)[0]
            for best_idx in best_pos:
                best_x_ = x[best_idx*dimension:(best_idx+1)*dimension]
                best_pos1.append(best_x_[0])
                best_pos2.append(best_x_[1])
            if m < best_obj:
                best_obj = m
                best_x = best_x_.copy()

    return best_x, best_obj, (pos1, pos2, best_pos1, best_pos2)


def main():
    bench_funcs = [
        ackley(), sphere(), rosenbrock(), styblinski(), k_tablet(),
        weighted_sphere(), different_power(), griewank()]

    dimension = 2
    num_population = 15
    max_iter = 30
    max_visit = 5
    for f in bench_funcs:
        print(f.name, "minimum =", f.opt)
        position, best, logs = optimize(
            dimension, num_population, max_visit, f, max_iter, 2)
        print("minimum is", best)
        print("position is", *position)
        f.plot(*logs, algo_name="paraABC")


if __name__ == '__main__':
    main()
