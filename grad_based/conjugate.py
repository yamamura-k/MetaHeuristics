"""
References :
- https://ja.wikipedia.org/wiki/%E5%85%B1%E5%BD%B9%E5%8B%BE%E9%85%8D%E6%B3%95
- https://ja.wikipedia.org/wiki/%E9%9D%9E%E7%B7%9A%E5%BD%A2%E5%85%B1%E5%BD%B9%E5%8B%BE%E9%85%8D%E6%B3%95
- 基礎数学 IV 最適化理論
"""
import numpy as np
from utils import lin_search, setup_logger
from utils.common import ContinuousOptResult

logger = setup_logger(__name__)


def getBeta(method, d, d_prev, s):
    if method == "FR":
        return np.float(d.T@d / d_prev.T@d_prev)
    elif method == "PR":
        return np.float(d.T@(d-d_prev) / d_prev.T@d_prev)
    elif method == "HS":
        return np.float(-d.T@(d-d_prev) / s.T@(d-d_prev))
    elif method == "DY":
        return np.float(-d.T@d / s.T@(d-d_prev))
    elif method == "heuristic":
        return max(0, np.float(d.T@(d-d_prev) / d_prev.T@d_prev))
    else:
        raise NotImplementedError


def optimize(x, objective, max_iter, method="exact", beta_method="FR", *args, **kwargs):
    try:
        objective.grad(x)
    except NotImplementedError:
        raise AttributeError(
            f"Gradient of {objective} is not defined.")
    result = ContinuousOptResult(objective, "CG", logger)
    result.post_process_per_iter(x, x, -1)

    d = -objective.grad(x)
    d_prev = d
    s = d
    alpha = lin_search(x, s, objective, method=method)
    for t in range(max_iter):
        x += alpha*s
        d = -objective.grad(x)
        beta = getBeta(beta_method, d, d_prev, s)
        s = beta*s + d
        alpha = lin_search(x, s, objective, method=method)
        d_prev = d

        result.post_process_per_iter(x, x, t)

    return result
