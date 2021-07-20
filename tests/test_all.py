from algorithm import optimize
from benchmarks import (ackley, different_power, griewank, k_tablet, log_exp,
                        rosenbrock, sphere, styblinski, weighted_sphere)



def test_all():
    dimension = 3
    num_population = 10
    max_iter = 10
    enable_grad = True
    if enable_grad:
        bench_funcs = [
            ackley(), sphere(), rosenbrock(), styblinski(dimension),
            k_tablet(dimension), weighted_sphere(dimension),
            different_power(dimension), griewank(dimension),
            log_exp(n=dimension)
        ]
    algorithms = ["paraABC", "paraBA", "ABC",
                  "BA", "GWO", "FA", "TLBO", "NM", "CG", "GD", "NV"]

    for algo in algorithms:
        for f in bench_funcs:
            options = dict(
                num_population=num_population,
                method="armijo",
                num_cpu=2,
                EXP=True,
                enable_grad=enable_grad,
                lb=f.boundaries[0],
                ub=f.boundaries[1],
                opt=f.opt,
                name=f.name,
            )
            optimize(dimension, f, max_iter, algo=algo, **options)
