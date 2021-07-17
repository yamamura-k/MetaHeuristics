import time

from algorithm import optimize
from benchmarks import (ackley, different_power, griewank, k_tablet,
                        rosenbrock, sphere, styblinski, weighted_sphere)
from utils import setup_logger


def main():

    dimension = 2
    num_population = 200
    max_iter = 100
    line_search = "armijo"
    num_cpu = 1
    EXP = False

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
    algorithms = ["paraABC", "paraBA", "ABC",
                  "BA", "GWO", "FA", "TLBO", "NM", "CG", "GD", "NV"][2:]
    L = len(bench_funcs)
    AL = len(algorithms)
    for algo in algorithms:
        if algo == "ABC" and num_population > 200:
            algo = "paraABC"
            num_cpu = 3
        print(algo, "is running ...")
        times = 0
        plot_time = 0
        for f in bench_funcs:
            options = dict(
                num_population=num_population,
                method=line_search,
                num_cpu=num_cpu,
                EXP=EXP,
                grad=f.grad,
                lb=f.boundaries[0],
                ub=f.boundaries[1],
                opt=f.opt,
                name=f.name,
            )
            stime = time.time()
            tmp = optimize(
                dimension, f, max_iter, algo=algo, **options)
            best = tmp.best_obj
            etime = time.time()
            result = f"| {f.name:55} | {f.opt:12.2f} | {best:12.2f} | {etime-stime:8.3f} | {algo:9}({tmp.num_restart}, {tmp.optimal})"
            results.append(result)
            times += etime - stime
            tmp.plot()
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
    logger = setup_logger.setLevel(20)
    main()
else:
    logger = setup_logger(__name__)
