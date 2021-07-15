import numpy as np

from utils import randomize, setup_logger
from utils.common import ResultManager

logger = setup_logger(__name__)


def optimize(dimension, objective, max_iter, alpha=1, gamma=2, rho=0.5, sigma=0.5, *args, **kwargs):
    x = randomize((dimension,), objective)
    best_x = None
    best_obj = np.inf

    x = np.vstack([x + np.eye(dimension), x])
    result = ResultManager(objective, "NM", logger, *args, **kwargs)
    for t in range(max_iter):
        obj_vals = np.array([objective(t) for t in x])
        orders = np.argsort(obj_vals)
        obj_vals = obj_vals[orders]
        x = x[orders]

        x_0 = np.mean(x[:-1], axis=0)
        x_worst = x[-1]

        x_f = x_0 + alpha * (x_0 - x_worst)
        obj_r = objective(x_f)

        obj_best = obj_vals[0]
        obj_second_worse = obj_vals[-2]
        obj_worst = obj_vals[-1]

        if obj_best < best_obj:
            best_obj = obj_best
            best_x = x[0].copy()

        if obj_best <= obj_r < obj_second_worse:
            x[-1] = x_f

        elif obj_r < obj_best:
            x_e = x_0 + gamma * (x_0 - x_worst)
            f_e = objective(x_e)

            if f_e < obj_r:
                x[-1] = x_e
                if f_e < best_obj:
                    best_obj = f_e
                    best_x = x_e.copy()
            else:
                x[-1] = x_f
                if obj_r < best_obj:
                    best_obj = obj_r
                    best_x = x_f.copy()

        elif obj_second_worse <= obj_r:
            x_c = x_0 + rho * (x_worst - x_0)
            f_c = objective(x_c)

            if f_c < obj_worst:
                x[-1] = x_c
            else:
                shrink_points = x[0] + \
                    sigma * (x - x[0])
                x[1:] = shrink_points[1:]

        if result.post_process_per_iter(x, best_x, t):
            break

    return result
