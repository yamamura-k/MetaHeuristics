import numpy as np


def optimize(x, objective, eps=1e-20, *args, **kwargs):
    try:
        objective.grad(np.zeros(2))
    except NotImplementedError:
        raise AttributeError(
            f"Gradient of {objective.__name__} is not defined.")
    try:
        objective.hesse(np.zeros(2))
    except NotImplementedError:
        raise AttributeError(
            f"Hesse matrix of {objective.__name__} is not defined.")

    nab = objective.grad(x)
    H_inv = np.linalg.inv(objective.hesse(x))
    lam = nab.T@H_inv@nab
    d = -H_inv@nab

    f_best = objective(x)
    x_best = x.copy()

    while lam > eps:
        eig, _ = np.linalg.eig(H_inv)
        assert (eig >= 0).all()

        x = x + d
        f = objective(x)
        nab = objective.grad(x)
        H_inv = np.linalg.inv(objective.hesse(x))
        lam = nab.T@H_inv@nab
        d = -H_inv@nab
        if f < f_best:
            f_best = f
            x_best = x.copy()

    return f_best, x_best
