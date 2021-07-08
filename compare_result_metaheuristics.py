import time

import nelder_mead as NM
from benchmarks import (ackley, different_power, griewank, k_tablet,
                        rosenbrock, sphere, styblinski, weighted_sphere)
from metaheuristics import ABC, BA, GWO
from utils import setup_logger

def main():
    dimension = 2
    num_population = 50
    max_iter = 20
    sep = "-"*112+"\n"
    sep_short = "-"*26+"\n"
    print(
        f"{sep_short}|  dim  |  pop  |  iter  |\n{sep_short}| {dimension:5} | {num_population:5} | {max_iter:6} |\n{sep_short}")
    results = []
    header = "".join(
        [sep, "| function", " "*47, " |    optimal   |   incumbent  | time[ms] | algorithm\n", sep])
    bench_funcs = [
        ackley(), sphere(), rosenbrock(), styblinski(dimension), k_tablet(),
        weighted_sphere(), different_power(), griewank()]
    algorithms = [ABC, BA, GWO, NM]
    L = len(bench_funcs)
    AL = len(algorithms)
    for algo in algorithms:
        print(algo.__name__, "is running ...")
        times = 0
        plot_time = 0
        for f in bench_funcs:
            stime = time.time()
            position, best, logs = algo.optimize(
                dimension, num_population, f, max_iter)
            etime = time.time()
            result = f"| {f.name:55} | {f.opt:12.2f} | {best:12.2f} | {etime-stime:8.3f} | {algo.__name__:9}"
            results.append(result)
            times += etime - stime
            if logs is None:
                continue
            # f.plot(*logs, algo_name=str(dimension)+"."+algo.__name__)
            plot_time += time.time() - etime
        print(
            "finish!", f"total {times:.3f} ms, average {times / L:.3f} ms, plot {plot_time:.3f} ms")
    results.sort(key=lambda x: x[2:5])

    print(header, end="")
    for i, line in enumerate(results):
        print(line)
        if i % AL == AL-1:
            print(sep, end="")


if __name__ == '__main__':
    logger = setup_logger.setLevel(0)
    main()
else:
    logger = setup_logger(__name__)