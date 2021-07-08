import numpy as np

from utils import setup_logger

logger = setup_logger(__name__)

def optimize(x, objective, max_iter, alpha=1e-4, *args, **kwargs):
    try:
        objective.grad(x)
    except NotImplementedError:
        raise AttributeError(
            f"Gradient of {objective.__name__} is not defined.")
    lam = 1
    lam_nx = None
    gam = -1
    y = x.copy()
    f_best = objective(x)
    x_best = x.copy()

    for t in range(max_iter):
        f = objective(x)
        if f < f_best:
            f_best = f
            x_best = x.copy()
        y_nx = x - objective.grad(x)*alpha
        x = y_nx + gam*(y_nx - y)
        y = y_nx
        lam_nx = 1 + np.sqrt(1+2*lam**2)/2
        gam = (lam - 1)/lam_nx
        lam = lam_nx
        logger.debug(f"iteration {t} [ best objective ] {f_best} [ step size ] {alpha}")

    return f_best, x_best
