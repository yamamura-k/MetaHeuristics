import numpy as np
from utils import getInitialPoint, setup_logger
from utils.common import ResultManager

logger = setup_logger(__name__)


def minimize(dimension, objective, eps=1e-10, *args, **kwargs):
    x = getInitialPoint((dimension,), objective)
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
    try:
        H_inv = np.linalg.inv(objective.hesse(x))
    except np.linalg.LinAlgError:
        logger.critical("Use pseudo inverse matrix.")
        H_inv = np.linalg.pinv(objective.hesse(x))
    lam = nab.T@H_inv@nab
    d = -H_inv@nab

    result = ResultManager(objective, __name__, logger, *args, **kwargs)
    result.post_process_per_iter(x, x, -1, lam=lam)
    if (np.isnan(nab)).any():
        logger.critical("gradient is nan.")
    t = 0
    while lam > eps:
        # eig, _ = np.linalg.eig(H_inv)
        # assert (eig >= 0).all()

        x = x + d
        nab = objective.grad(x)
        try:
            H_inv = np.linalg.inv(objective.hesse(x))
        except np.linalg.LinAlgError:
            logger.critical("Use pseudo inverse matrix.")
            H_inv = np.linalg.pinv(objective.hesse(x))
        lam = nab.T@H_inv@nab
        d = -H_inv@nab
        if result.post_process_per_iter(x, x, t, lam=lam, grad=nab):
            break
        t += 1

    return result
