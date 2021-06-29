"""
Sample implementation of Artificial Bee Colony Algorithm.

Reference : https://link.springer.com/content/pdf/10.1007/s10898-007-9149-x.pdf
"""
import numpy as np

from benchmarks import (ackley, different_power, griewank, k_tablet,
                        rosenbrock, sphere, styblinski, weighted_sphere)


def optimize(dimension, num_population, max_visit, f, C):
    # step1 : initialization
    xs = np.random.random((num_population, dimension))
    all_candidates = np.arange(num_population)
    v = np.array([f(x) for x in xs])
    cnt = np.zeros(num_population)

    def update(i):
        x_i = xs[i].copy()
        j = np.random.randint(0, dimension-1)
        k = np.random.randint(0, num_population-1)
        phi = np.random.normal()
        x_i[j] -= phi*(x_i[j] - xs[k][j])
        v_new = f(x_i)
        if v_new <= v[i]:
            xs[i] = x_i
            v[i] = v_new
        cnt[i] += 1

    def random_update():
        candidate = np.where(cnt == max_visit)[0]
        for i in candidate:
            x_i = np.random.random(dimension)
            v_new = f(x_i)
            if v_new <= v[i]:
                xs[i] = x_i
                v[i] = v_new
                cnt[i] = 1
    pos1 = []
    pos2 = []
    best_pos1 = []
    best_pos2 = []

    for c in range(1, C+1):
        for _ in range(num_population):
            # employed bees
            i = np.random.randint(0, num_population-1)
            update(i)

            # onlooker bees
            if (v >= 0).all():
                probs = v / np.sum(v)
            else:
                m = np.min(v)
                w = v - m
                probs = w / np.sum(w)
            probs = 1 - probs
            probs /= np.sum(probs)
            i = np.random.choice(all_candidates, p=probs)
            update(i)

            # scouts
            random_update()

        m = np.min(v)
        pos1.append([x[0] for x in xs])
        pos2.append([x[1] for x in xs])
        best_pos = np.where(v == m)[0]
        for idx in best_pos:
            best_pos1.append(xs[idx][0])
            best_pos2.append(xs[idx][1])

    min_idx = np.where(v == np.min(v))[0][0]

    return xs[min_idx], v[min_idx], (pos1, pos2, best_pos1, best_pos2)


def main():
    bench_funcs = [
        ackley(), sphere(), rosenbrock(), styblinski(), k_tablet(),
        weighted_sphere(), different_power(), griewank()]

    dimension = 2
    num_population = 15
    max_iter = 30
    max_visit = 5
    for f in bench_funcs:
        print(f.name, "minimum =", f.opt)
        position, best, logs = optimize(
            dimension, num_population, max_visit, f, max_iter)
        print("minimum is", best)
        print("position is", *position)
        f.plot(*logs, algo_name="ABC")


if __name__ == '__main__':
    main()
