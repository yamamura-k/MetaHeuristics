

import nelder_mead as NM
from benchmarks import (ackley, different_power, griewank, k_tablet,
                        rosenbrock, sphere, styblinski, weighted_sphere)
from metaheuristics import ABC, BA, GWO, paraABC, paraBA
from utils.grad_based import check_grad

dimension = 2
num_population = 10
max_iter = 20
bench_funcs = [
    ackley(), sphere(), rosenbrock(), styblinski(dimension), k_tablet(),
    weighted_sphere(), different_power(), griewank()]


def test_ABC():
    for f in bench_funcs:
        result = ABC.optimize(
            dimension, f, max_iter, num_population=num_population)
        check_grad(result.best_x, f)


def test_paraABC():
    for f in bench_funcs:
        result = paraABC.optimize(
            dimension, f, max_iter, num_population=num_population, num_cpu=1)


def test_BA():
    for f in bench_funcs:
        result = BA.optimize(
            dimension, f, max_iter, num_population=num_population)


def test_GWO():
    for f in bench_funcs:
        result = GWO.optimize(
            dimension, f, max_iter, num_population=num_population)


def test_NM():
    for f in bench_funcs:
        result = NM.optimize(
            dimension, f, max_iter, num_population=num_population)


def test_paraBA():
    for f in bench_funcs:
        result = paraBA.optimize(
            dimension, f, max_iter, num_population=num_population, num_cpu=1)
