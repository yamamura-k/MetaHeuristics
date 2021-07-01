import numpy as np
from utils import calc_stepsize


def optimize(x, objective, max_iter, alpha=1e-4,
             method="static", *args, **kwargs):
    try:
        objective.grad(np.zeros(2))
    except NotImplementedError:
        raise AttributeError(
            f"Gradient of {objective.__name__} is not defined.")
    f_best = objective(x)
    x_best = x.copy()

    for k in range(max_iter):
        alpha = calc_stepsize(method, x, alpha, objective)
        if not np.isscalar(alpha):
            print(method)
            raise AssertionError
        f = objective(x)
        if f < f_best:
            f_best = f
            x_best = x.copy()
        x -= alpha*objective.grad(x)

    return f_best, x_best
