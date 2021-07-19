import numpy as np
from jax import device_put
from utils import getInitialPoint, lin_search, setup_logger
from utils.common import ResultManager

logger = setup_logger(__name__)


def minimize(dimension, objective, max_iter, alpha=1e-4, method="exact", *args, **kwargs):
    x = getInitialPoint((dimension,), objective)
    try:
        objective.grad(device_put(x)).block_until_ready()
    except NotImplementedError:
        raise AttributeError(
            f"Gradient of {objective} is not defined.")
    lam = 1
    lam_nx = None
    gam = -1
    y = x.copy()
    result = ResultManager(objective, __name__, logger, *args, **kwargs)
    result.post_process_per_iter(x, x, -1)
    d_prev = -objective.grad(device_put(x)).block_until_ready()

    for t in range(max_iter):
        if alpha == 0:
            break
        y_nx = x + d_prev*alpha
        x = y_nx + gam*(y_nx - y)
        d = -objective.grad(device_put(x)).block_until_ready()

        y = y_nx
        lam_nx = 1 + np.sqrt(1+2*lam**2)/2
        gam = (lam - 1)/lam_nx
        lam = lam_nx

        alpha = lin_search((1+gam)*x - gam*y, (1+gam) *
                           d, objective, alpha=alpha, method=method)
        if result.post_process_per_iter(x, x, t, alpha=alpha, grad=-d):
            break
        d_prev = d
    return result
