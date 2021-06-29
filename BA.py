import numpy as np

from benchmarks import (ackley, different_power, griewank, k_tablet,
                        rosenbrock, sphere, styblinski, weighted_sphere)


def optimize(dimension, num_populatioin, objective, max_iter, f_min=0,
             f_max=100, selection_max=10, alpha=0.9, gamma=0.9):
    x = np.random.random((num_populatioin, dimension))
    v = np.random.random((num_populatioin, dimension))
    f = np.random.uniform(f_min, f_max, size=num_populatioin)
    A = np.random.uniform(1, 2, size=num_populatioin)
    r = np.random.uniform(0, 1, size=num_populatioin)
    r0 = r.copy()

    obj_best = float('inf')
    best_x = None
    for i in range(num_populatioin):
        obj_tmp = objective(x[i])
        if obj_tmp < obj_best:
            obj_best = obj_tmp
            best_x = x[i].copy()

    pos1 = []
    pos2 = []
    best_pos1 = []
    best_pos2 = []

    for step in range(max_iter):
        for i in range(num_populatioin):
            f[i] = f_min + (f_max - f_min)*np.random.uniform(0, 1)
            v[i] += (x[i] - best_x) * f[i]
            x_t = x[i] + v[i]
            obj_new = None
            obj_t = objective(x_t)

            if np.random.rand() > r[i]:
                eps = np.random.uniform(-1, 1)
                idx = np.random.randint(0, selection_max)
                x_new = x[idx].copy()
                x_new += eps * np.average(A)
                obj_new = objective(x_new)

            x_random = np.random.random(dimension)
            obj_random = objective(x_random)
            if (obj_new is None or obj_t > obj_new) and obj_t > obj_random:
                if obj_new is None or obj_new <= obj_random:
                    x[i] = x_random
                    if obj_random < obj_best:
                        obj_best = obj_random
                        best_x = x[i].copy()
                else:
                    x[i] = x_new
                    if obj_new < obj_best:
                        obj_best = obj_new
                        best_x = x[i].copy()
            else:
                x[i] = x_t
                if obj_t < obj_best:
                    obj_best = obj_t
                    best_x = x[i].copy()

            r[i] = r0[i] * (1-np.exp(-gamma*step))
            A[i] *= alpha
            x = list(x)
            x.sort(key=lambda s: objective(s))
            x = np.array(x)

        pos1.append([x[0] for x in x])
        pos2.append([x[1] for x in x])
        best_pos1.append(best_x[0])
        best_pos2.append(best_x[1])

    return best_x, obj_best, (pos1, pos2, best_pos1, best_pos2)


def main():
    bench_funcs = [
        ackley(), sphere(), rosenbrock(), styblinski(), k_tablet(),
        weighted_sphere(), different_power(), griewank()]

    dimension = 2
    num_population = 50
    max_iter = 100
    for f in bench_funcs:
        print(f.name, "minimum =", f.opt)
        position, best, logs = optimize(
            dimension, num_population, f, max_iter)
        print("minimum is", best)
        print("position is", *position)
        f.plot(*logs, algo_name="BA")


if __name__ == '__main__':
    main()
