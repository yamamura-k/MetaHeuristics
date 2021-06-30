import ABC
import BA
import paraABC
import paraBA
from benchmarks import (ackley, different_power, griewank, k_tablet,
                        rosenbrock, sphere, styblinski, weighted_sphere)


def main():
    bench_funcs = [
        ackley(), sphere(), rosenbrock(), styblinski(), k_tablet(),
        weighted_sphere(), different_power(), griewank()]
    algorithms = [ABC, paraABC, BA, paraBA]

    dimension = 2
    num_population = 15
    max_iter = 30
    for algo in algorithms:
        for f in bench_funcs:
            position, best, logs = algo.optimize(
                dimension, num_population, f, max_iter)
            print(f.name, "opt - best =", f.opt - best)
            print("position is", *position)
            f.plot(*logs, algo_name="BA")


if __name__ == '__main__':
    main()
