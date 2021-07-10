import numpy as np

from utils import setup_logger

logger = setup_logger(__name__)


def optimize(x, objective, eps=1e-20, *args, **kwargs):
    try:
        objective.grad(x)
    except NotImplementedError:
        raise AttributeError(
            f"Gradient of {objective} is not defined.")
    try:
        objective.hesse(x)
    except NotImplementedError:
        raise AttributeError(
            f"Hesse matrix of {objective} is not defined.")

    nab = objective.grad(x)
    H_inv = np.linalg.inv(objective.hesse(x))
    lam = nab.T@H_inv@nab
    d = -H_inv@nab

    f_best = objective(x)
    x_best = x.copy()
    t = 0
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
        logger.debug(f"iteration {t} [ best objective ] {f_best}")
        t += 1

    return f_best, x_best
