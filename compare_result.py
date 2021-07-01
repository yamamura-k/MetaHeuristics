import time

import ABC
import BA
import paraABC
import paraBA
from benchmarks import (ackley, different_power, griewank, k_tablet,
                        rosenbrock, sphere, styblinski, weighted_sphere)


def main():
    dimension = 2
    num_population = 100
    max_iter = 20
    sep = "-"*102+"\n"
    sep_short = "-"*26+"\n"
    print(
        f"{sep_short}|  dim  |  pop  |  iter  |\n{sep_short}| {dimension:5} | {num_population:5} | {max_iter:6} |\n{sep_short}")
    results = []
    header = "".join(
        [sep, "| function", " "*47, " |   optimal   |  incumbent  | time[ms] | algorithm |\n", sep])
    bench_funcs = [
        ackley(), sphere(), rosenbrock(), styblinski(dimension), k_tablet(),
        weighted_sphere(), different_power(), griewank()]
    algorithms = [ABC, paraABC, BA, paraBA]

    for algo in algorithms:
        print(algo.__name__, "is running ...")
        for f in bench_funcs:
            stime = time.time()
            position, best, logs = algo.optimize(
                dimension, num_population, f, max_iter)
            etime = time.time()
            result = f"| {f.name:55} | {f.opt:12.2f} | {best:12.2f} | {etime-stime:8.3f} | {algo.__name__:9} |\n"
            results.append(result)
            # f.plot(*logs, algo_name=algo.__name__)
    results.sort(key=lambda x: x[2])
    print(algo.__name__, "finish!")

    print(header, end="")
    for i, line in enumerate(results):
        print(line, end="")
        if i % 4 == 3:
            print(sep, end="")


if __name__ == '__main__':
    main()
