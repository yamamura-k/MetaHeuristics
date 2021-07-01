import time

import numpy as np

import grad_based.conjugate as CG
import grad_based.gradient_descent as GD
import grad_based.nesterov as NV
import grad_based.newton as NW
from benchmarks import log_exp, pow


def main():
    n, m = 10, 100
    x = np.random.randn(n)
    max_iter = 100
    bench_funcs = [log_exp(n=n, m=m), pow()]
    algorithms = [GD, CG, NV, NW]

    for bench_func in bench_funcs:
        for algo in algorithms:
            print(algo.__name__, "is running ...")
            best_f, best_x = algo.optimize(x.copy(), bench_func, max_iter)
            print(best_f)


if __name__ == '__main__':
    main()
