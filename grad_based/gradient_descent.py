import numpy as np
from utils import lin_search, randomize, setup_logger
from utils.common import FunctionWrapper, ResultManager

logger = setup_logger(__name__)


def optimize(dimension, objective, max_iter, alpha=1e-4,
             method="exact", *args, **kwargs):
    objective = FunctionWrapper(objective, *args, **kwargs)
    x = randomize((dimension, 1), objective)
    try:
        objective.grad(x)
    except NotImplementedError:
        raise AttributeError(
            f"Gradient of {objective} is not defined.")

    result = ResultManager(objective, "GD", logger, *args, **kwargs)
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
        if result.post_process_per_iter(x, x, t):
            break

    return result
