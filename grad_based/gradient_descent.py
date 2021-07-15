import numpy as np
from utils import lin_search, setup_logger
from utils.common import ContinuousOptResult

logger = setup_logger(__name__)


def optimize(x, objective, max_iter, alpha=1e-4,
             method="exact", *args, **kwargs):
    try:
        objective.grad(x)
    except NotImplementedError:
        raise AttributeError(
            f"Gradient of {objective} is not defined.")

    result = ContinuousOptResult(objective, "GD", logger)
    result.post_process_per_iter(x, x, -1)

    for t in range(max_iter):
        alpha = lin_search(x, -objective.grad(x), objective,
                           alpha=alpha, method=method)
        if alpha == 0:
            break
        if not np.isscalar(alpha):
            print(method)
            raise AssertionError
        x -= alpha*objective.grad(x)
        result.post_process_per_iter(x, x, t)

    return result
