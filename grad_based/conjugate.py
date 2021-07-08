"""
References :
- https://ja.wikipedia.org/wiki/%E5%85%B1%E5%BD%B9%E5%8B%BE%E9%85%8D%E6%B3%95
- https://ja.wikipedia.org/wiki/%E9%9D%9E%E7%B7%9A%E5%BD%A2%E5%85%B1%E5%BD%B9%E5%8B%BE%E9%85%8D%E6%B3%95
- 基礎数学 IV 最適化理論
"""
import numpy as np
from scipy.optimize import minimize_scalar
from utils import setup_logger

logger = setup_logger(__name__)

def calc_beta(method, d, d_prev, s):
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


def lin_search(x, objective, s):
    def phi(alpha):
        return objective(x-alpha*s)
    return np.float(minimize_scalar(phi).x)


def optimize(x, objective, max_iter, method="FR", *args, **kwargs):
    try:
        objective.grad(x)
    except NotImplementedError:
        raise AttributeError(
            f"Gradient of {objective.__name__} is not defined.")
    f_best = objective(x)
    x_best = x.copy()
    d = -objective.grad(x)
    d_prev = d
    s = d
    alpha = lin_search(x, objective, s)
    for t in range(max_iter):
        x += alpha*s
        d = -objective.grad(x)
        beta = calc_beta(method, d, d_prev, s)
        s = beta*s + d
        alpha = lin_search(x, objective, s)
        d_prev = d
        f = objective(x)
        if f < f_best:
            f_best = f
            x_best = x.copy()
        logger.debug(f"iteration {t} [ best objective ] {f_best} [ beta ] {beta}")
    return f_best, x_best
