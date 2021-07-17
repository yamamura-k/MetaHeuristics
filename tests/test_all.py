from algorithm import optimize
from benchmarks import (ackley, different_power, griewank, k_tablet,
                        rosenbrock, sphere, styblinski, weighted_sphere)


def test_all():
    dimension = 3
    num_population = 10
    max_iter = 10
    bench_funcs = [
        ackley(), sphere(), rosenbrock(), styblinski(dimension), k_tablet(),
        weighted_sphere(), different_power(), griewank()]
    algorithms = ["paraABC", "paraBA", "ABC",
                  "BA", "GWO", "FA", "TLBO", "NM", "CG", "GD", "NV"]

    for algo in algorithms:
        for f in bench_funcs:
            options = dict(
                num_population=num_population,
                method="armijo",
                num_cpu=2,
                EXP=True,
                grad=f.grad,
                lb=f.boundaries[0],
                ub=f.boundaries[1],
                opt=f.opt,
                name=f.name,
            )
            optimize(dimension, f, max_iter, algo=algo, **options)
