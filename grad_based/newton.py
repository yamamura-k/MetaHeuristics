import numpy as np
from utils import setup_logger
from utils.common import ContinuousOptResult

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

    result = ContinuousOptResult(objective, "NW", logger)
    result.post_process_per_iter(x, x, -1)

    t = 0
    while lam > eps:
        eig, _ = np.linalg.eig(H_inv)
        assert (eig >= 0).all()

        x = x + d
        nab = objective.grad(x)
        H_inv = np.linalg.inv(objective.hesse(x))
        lam = nab.T@H_inv@nab
        d = -H_inv@nab
        result.post_process_per_iter(x, x, t)
        t += 1

    return result
