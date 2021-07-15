import numpy as np
from utils import randomize, setup_logger
from utils.common import ContinuousOptResult

logger = setup_logger(__name__)

# numpy version


def optimize(dimension, num_population, objective, max_iter, f_min=0,
             f_max=100, selection_max=10, alpha=0.9, gamma=0.9):
    x = randomize((num_population, dimension), objective)
    v = np.random.random((num_population, dimension))
    f = np.random.uniform(f_min, f_max, size=num_population)
    A = np.random.uniform(1, 2, size=num_population)
    r = np.random.uniform(0, 1, size=num_population)
    r0 = r.copy()

    obj_best = np.inf
    best_x = None
    for i in range(num_population):
        obj_tmp = objective(x[i])
        if obj_tmp < obj_best:
            obj_best = obj_tmp
            best_x = x[i].copy()

    result = ContinuousOptResult(objective, "BA", logger)
    result.post_process_per_iter(x, best_x, -1)

    for step in range(max_iter):
        obj_current = np.array([objective(t) for t in x])
        f = f_min + (f_max - f_min) * \
            np.broadcast_to(np.random.uniform(
                0, 1, size=num_population), (dimension, num_population)).T
        v += (x - np.broadcast_to(best_x, x.shape)) * f
        x_t = x + v
        obj_t = np.array([objective(t) for t in x_t])
        obj_new = np.full(num_population, np.inf)

        idxs = np.where(np.random.rand(*r.shape) > r)
        x_new = np.empty_like(x)
        idx = np.random.randint(0, selection_max, size=(len(idxs[0]),))
        eps = np.broadcast_to(
            np.random.uniform(-1, 1, size=(len(idxs[0]),)), (dimension, len(idxs[0]))).T
        x_new[idxs] = x[idx] + eps*np.average(A)
        obj_new[idxs] = np.array([objective(x_new[t]) for t in idxs[0]])

        x_random = randomize((num_population, dimension), objective)
        obj_random = np.array([objective(t) for t in x_random])

        idxs1 = np.where(((obj_new == np.inf) | (obj_t > obj_new)) & (
            obj_t > obj_random) & (obj_random > obj_new))
        x[idxs1] = x_new[idxs1]
        obj_current[idxs1] = obj_new[idxs1]

        idxs2 = np.where(((obj_new == np.inf) | (obj_t > obj_new)) & (
            obj_t > obj_random) & (~(obj_random > obj_new)))
        x[idxs2] = x_random[idxs2]
        obj_current[idxs2] = obj_random[idxs2]

        idxs3 = np.where(~(((obj_new == np.inf) | (obj_t > obj_new)) & (
            obj_t > obj_random)))
        x[idxs3] = x_t[idxs3]
        obj_current[idxs3] = obj_t[idxs3]

        obj_tmp = np.min(obj_current)
        if obj_tmp < obj_best:
            obj_best = obj_tmp
            best_x = x[np.where(obj_current == obj_tmp)][0].copy()

        r = r0 * (1-np.exp(-gamma*step))
        A *= alpha

        result.post_process_per_iter(x, best_x, step)

    return result

# slower version


def _optimize(dimension, num_population, objective, max_iter, f_min=0,
              f_max=100, selection_max=10, alpha=0.9, gamma=0.9):
    x = randomize((num_population, dimension), objective)
    v = np.random.random((num_population, dimension))
    f = np.random.uniform(f_min, f_max, size=num_population)
    A = np.random.uniform(1, 2, size=num_population)
    r = np.random.uniform(0, 1, size=num_population)
    r0 = r.copy()

    obj_best = np.inf
    best_x = None
    for i in range(num_population):
        obj_tmp = objective(x[i])
        if obj_tmp < obj_best:
            obj_best = obj_tmp
            best_x = x[i].copy()

    result = ContinuousOptResult(objective, "BA", logger)
    result.post_process_per_iter(x, best_x, -1)

    for step in range(max_iter):
        for i in range(num_population):
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

            x_random = randomize((dimension, ), objective)
            obj_random = objective(x_random)
            if (obj_new is None or obj_t > obj_new) and obj_t > obj_random:
                if obj_new is None or obj_new >= obj_random:
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

        result.post_process_per_iter(x, best_x, step)

    return result
