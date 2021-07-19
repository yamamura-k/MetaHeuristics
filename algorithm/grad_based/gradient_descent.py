import numpy as np
from jax import device_put
from utils import getInitialPoint, lin_search, setup_logger
from utils.common import ResultManager

logger = setup_logger(__name__)


def minimize(dimension, objective, max_iter, alpha=1e-4,
             method="exact", *args, **kwargs):
    x = getInitialPoint((dimension,), objective)
    try:
        objective.grad(device_put(x)).block_until_ready()
    except NotImplementedError:
        raise AttributeError(
            f"Gradient of {objective} is not defined.")

    result = ResultManager(objective, __name__, logger, *args, **kwargs)
    result.post_process_per_iter(x, x, -1)

    for t in range(max_iter):
        alpha = lin_search(x, -objective.grad(device_put(x)).block_until_ready(), objective,
                           alpha=alpha, method=method)
        if alpha == 0:
            break
        if not np.isscalar(alpha):
            print(method)
            raise AssertionError
        d = -objective.grad(device_put(x)).block_until_ready()
        x += alpha*d
        if result.post_process_per_iter(x, x, t, alpha=alpha, grad=-d):
            break

    return result
