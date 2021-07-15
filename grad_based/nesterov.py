import numpy as np
from utils import lin_search, setup_logger
from utils.common import ContinuousOptResult

logger = setup_logger(__name__)


def optimize(x, objective, max_iter, alpha=1e-4, method="exact", *args, **kwargs):
    try:
        objective.grad(x)
    except NotImplementedError:
        raise AttributeError(
            f"Gradient of {objective} is not defined.")
    lam = 1
    lam_nx = None
    gam = -1
    y = x.copy()
    result = ContinuousOptResult(objective, "NV", logger)
    result.post_process_per_iter(x, x, -1)

    for t in range(max_iter):
        if alpha == 0:
            break
        y_nx = x - objective.grad(x)*alpha
        x = y_nx + gam*(y_nx - y)
        y = y_nx
        lam_nx = 1 + np.sqrt(1+2*lam**2)/2
        gam = (lam - 1)/lam_nx
        lam = lam_nx

        alpha = lin_search((1+gam)*x - gam*y, -(1+gam) *
                           objective.grad(x), objective, alpha=alpha, method=method)
        result.post_process_per_iter(x, x, t)

    return result
