

from benchmarks import (ackley, different_power, griewank, k_tablet,
                        rosenbrock, sphere, styblinski, weighted_sphere)
from metaheuristics import ABC, BA, paraABC, paraBA

dimension = 2
num_population = 100
max_iter = 20
bench_funcs = [
    ackley(), sphere(), rosenbrock(), styblinski(dimension), k_tablet(),
    weighted_sphere(), different_power(), griewank()]


def test_ABC():
    for f in bench_funcs:
        position, best, logs = ABC.optimize(
            dimension, num_population, f, max_iter)
    f.plot(*logs, algo_name=ABC.__name__)


def test_paraABC():
    for f in bench_funcs:
        position, best, logs = paraABC.optimize(
            dimension, num_population, f, max_iter)


def test_BA():
    for f in bench_funcs:
        position, best, logs = BA.optimize(
            dimension, num_population, f, max_iter)


def test_paraBA():
    for f in bench_funcs:
        position, best, logs = paraBA.optimize(
            dimension, num_population, f, max_iter)
