import time

from _benchmarks import (ackley_without_grad, different_power_without_grad,
                         griewank_without_grad, k_tablet_without_grad,
                         log_exp_without_grad, rosenbrock_without_grad,
                         sphere_without_grad, styblinski_without_grad,
                         weighted_sphere_without_grad)
from algorithm import optimize
from benchmarks import (ackley, different_power, griewank, k_tablet, log_exp,
                        rosenbrock, sphere, styblinski, weighted_sphere)
from utils import getBestParams, setup_logger, update_params

dimension = 20
enable_grad = True
if enable_grad:
    bench_funcs = [
        ackley(), sphere(), rosenbrock(), styblinski(dimension),
        k_tablet(dimension), weighted_sphere(dimension),
        different_power(dimension), griewank(dimension),
        log_exp(n=dimension)
    ]
else:
    bench_funcs = [
        ackley_without_grad(), sphere_without_grad(
        ), rosenbrock_without_grad(), styblinski_without_grad(dimension),
        k_tablet_without_grad(
            dimension), weighted_sphere_without_grad(dimension),
        different_power_without_grad(
            dimension), griewank_without_grad(dimension),
        log_exp_without_grad(n=dimension)
    ]

algorithms = ["paraABC", "paraBA", "ABC",
              "BA", "GWO", "FA", "TLBO", "NM", "CG", "GD", "NV", "NW"][2:]


def hypara_opt(num_processes=2, n_jobs=2):
    from multiprocessing import Pool
    with Pool(processes=num_processes) as p:
        inputs = [(dimension, f, algo, n_jobs)
                  for f in bench_funcs for algo in algorithms]
        p.starmap(getBestParams, inputs)


def main():
    num_population = 200
    max_iter = 100
    line_search = "armijo"
    num_cpu = 1
    EXP = True

    sep = "-"*112+"\n"
    sep_short = "-"*26+"\n"
    print(
        f"{sep_short}|  dim  |  pop  |  iter  |\n{sep_short}| {dimension:5} | {num_population:5} | {max_iter:6} |\n{sep_short}")
    results = []
    header = "".join(
        [sep, "| function", " "*47, " |    optimal   |   incumbent  | time[ms] | algorithm\n", sep])
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
                max_iter=max_iter,
                num_population=num_population,
                method=line_search,
                num_cpu=num_cpu,
                EXP=EXP,
                enable_grad=enable_grad,
                # grad=f.grad,
                lb=f.boundaries[0],
                ub=f.boundaries[1],
                opt=f.opt,
                name=f.name,
            )
            param = getBestParams(dimension, f, algo, is_search=False)
            options = update_params(options, param)
            print(options)
            stime = time.time()
            tmp = optimize(
                dimension, f, algo=algo, **options)
            best = tmp.best_obj
            etime = time.time()
            result = f"| {f.name:55} | {f.opt:12.2f} | {best:12.2f} | {etime-stime:8.3f} | {algo:9}({tmp.num_restart}, {tmp.optimal})"
            results.append(result)
            times += etime - stime
            # tmp.plot()
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
    # hypara_opt()
    main()
else:
    logger = setup_logger(__name__)
