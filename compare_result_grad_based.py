import numpy as np

from benchmarks import (ackley, different_power, griewank, k_tablet, log_exp,
                        rosenbrock, sphere, styblinski, weighted_sphere)
from grad_based import CG, GD, NV, NW
from utils import check_grad, setup_logger


def main():
    n, m = 2, 100
    x = np.random.random((n, 1))
    # x = np.zeros((n, 1))
    max_iter = 100
    bench_funcs = []
    bench_funcs = [
        log_exp(n=n, m=m), ackley(), sphere(), rosenbrock(
        ), styblinski(n), k_tablet(),
        weighted_sphere(), different_power(), griewank()]
    algorithms = [GD, CG, NV, NW]
    result = []

    for bench_func in bench_funcs:
        print("\n", bench_func.name)
        result.append([(bench_func.name, ), ])
        for algo in algorithms:
            if algo == NW:
                continue
            for m in ["exact", "armijo"]:
                best_f, best_x = algo.optimize(
                    x.copy(), bench_func, max_iter, method=m)
                best_f = np.asscalar(best_f)
                result.append([(algo.__name__, best_f), best_x])
    for r in result:
        print(*r[0])
        if len(r) > 1:
            check_grad(r[1], bench_func)


if __name__ == '__main__':
    logger = setup_logger.setLevel(0)
    main()
else:
    logger = setup_logger(__name__)
