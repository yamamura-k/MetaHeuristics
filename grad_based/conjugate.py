"""
References :
- https://ja.wikipedia.org/wiki/%E5%85%B1%E5%BD%B9%E5%8B%BE%E9%85%8D%E6%B3%95
- https://ja.wikipedia.org/wiki/%E9%9D%9E%E7%B7%9A%E5%BD%A2%E5%85%B1%E5%BD%B9%E5%8B%BE%E9%85%8D%E6%B3%95
- 基礎数学 IV 最適化理論
"""
import numpy as np
from scipy.optimize import minimize_scalar


def calc_beta(method, d, d_prev, s):
    if method == "fletcher-reeves":
        return np.float(d.T@d / d_prev.T@d_prev)
    elif method == "polak–ribiere":
        return np.float().T@(d-d_prev) / d_prev.T@d_prev
    elif method == "hestenes-stiefel":
        return np.float(-d.T@(d-d_prev) / s.T@(d-d_prev))
    elif method == "dai–yuan":
        return np.float(-d.T@d / s.T@(d-d_prev))
    elif method == "heuristic":
        return max(0, np.float(d.T@(d-d_prev) / d_prev.T@d_prev))
    else:
        raise NotImplementedError


def lin_search(x, objective, s):
    def phi(alpha):
        return objective(x-alpha*s)
    return np.float(minimize_scalar(phi).x)


def optimize(x, objective, max_iter, method="fletcher-reeves", *args, **kwargs):
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
    for _ in range(max_iter):
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
    return f_best, x_best
