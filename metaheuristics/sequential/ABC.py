"""
Sample implementation of Artificial Bee Colony Algorithm.

Reference : https://link.springer.com/content/pdf/10.1007/s10898-007-9149-x.pdf
"""
import numpy as np
from utils import randomize, setup_logger
from utils.common import ContinuousOptResult

logger = setup_logger(__name__)


def optimize(dimension, num_population, objective, max_iter, max_visit=10):
    # step1 : initialization
    x = randomize((num_population, dimension), objective)
    all_candidates = np.arange(num_population)
    v = np.array([objective(t) for t in x])
    cnt = np.zeros(num_population)

    def update(i):
        x_i = x[i].copy()
        j = np.random.randint(0, dimension-1)
        k = np.random.randint(0, num_population-1)
        phi = np.random.normal()
        x_i[j] -= phi*(x_i[j] - x[k][j])
        v_new = objective(x_i)
        if v_new <= v[i]:
            x[i] = x_i
            v[i] = v_new
        cnt[i] += 1

    def random_update():
        candidate = np.where(cnt == max_visit)[0]
        for i in candidate:
            x_i = randomize((dimension, ), objective)
            v_new = objective(x_i)
            if v_new <= v[i]:
                x[i] = x_i
                v[i] = v_new
                cnt[i] = 1

    result = ContinuousOptResult(objective, "ABC", logger)
    m = np.min(v)
    best_pos = np.where(v == m)
    result.post_process_per_iter(x, x[best_pos][0], -1)

    for t in range(1, max_iter+1):
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
        best_pos = np.where(v == m)
        result.post_process_per_iter(x, x[best_pos][0], t)

    return result
