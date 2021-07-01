import time

import numpy as np

import grad_based.conjugate as CG
import grad_based.gradient_descent as GD
import grad_based.nesterov as NV
import grad_based.newton as NW
from benchmarks import log_exp


def main():
    n, m = 10, 100
    x = np.random.randn(n, 1)
    max_iter = 100
    bench_funcs = [log_exp(n=n, m=m)]
    algorithms = [GD, CG, NV, NW]

    for bench_func in bench_funcs:
        print("\n", bench_func.name)
        for algo in algorithms:
            best_f, best_x = algo.optimize(x.copy(), bench_func, max_iter)
            print(algo.__name__, best_f)


if __name__ == '__main__':
    main()
