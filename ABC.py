"""
Sample implementation of Artificial Bee Colony Algorithm.

Reference : https://link.springer.com/content/pdf/10.1007/s10898-007-9149-x.pdf
"""
import numpy as np
from benchmarks import ackley, sphere, rosenbrock


def optimize(dimension, num_population, f, C):
    # step1 : initialization
    xs = np.random.random((num_population, dimension))
    all_candidates = np.arange(num_population)
    v = [f(x) for x in xs]

    def update(i):
        x_i = xs[i].copy()
        j = np.random.randint(0, dimension-1)
        k = np.random.randint(0, num_population-1)
        phi = np.random.normal()
        x_i[j] -= phi*(x_i[j] - xs[k][j])
        v_new = f(x_i)
        if v_new <= v[i]:
            xs[i] = v_new

    def random_update():
        i = np.random.randint(0, num_population-1)
        x_i = np.random.random((1, dimension))
        v_new = f(x_i)
        if v_new <= v[i]:
            xs[i] = v_new

    for c in range(1, C+1):
        # employed bees
        i = np.random.randint(0, num_population-1)
        update(i)
        # onlooker bees
        i = np.random.choice(all_candidates, p=v/np.sum(v))
        update(i)
        # scouts
        random_update()
    min_idx = v.index(np.min(v))
    print("minimum is", v[min_idx])
    return xs[min_idx], v[min_idx]


def main():
    for f in [ackley(), sphere(), rosenbrock()]:
        print(f.name, "minimum =", f.opt)
        print("position is", *optimize(2, 1000, f, 10000)[0])


if __name__=='__main__':
    main()