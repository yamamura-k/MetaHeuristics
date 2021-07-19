import numpy as np
from scipy.optimize import minimize_scalar


def lin_search(x, d, objective, alpha=None, method="exact"):
    if method == "wolfe":
        return wolfe(x, d, objective)
    elif method == "armijo":
        return armijo(x, d, objective)
    elif method == "static":
        return alpha
    elif method == "exact":
        def phi(alpha):
            return objective(x+alpha*d)
        return np.float(minimize_scalar(phi).x)
    else:
        raise NotImplementedError


def wolfe(x, d, objective, c1=1e-5, c2=1-1e-5, alpha=10):
    a = 0
    b = np.inf
    nab = objective.grad(device_put(x)).block_until_ready()
    f = objective(x)
    phi_dif0 = np.dot(nab.T, d)
    assert phi_dif0 <= 0
    prev_alpha = alpha
    while True:
        phi = objective(x+alpha*d)
        phi_dif = np.dot(objective.grad(x+alpha*d).T, d)
        if phi > f + c1*alpha*phi_dif0:
            b = alpha
        elif phi_dif < c2*phi_dif0:
            a = alpha
        else:
            return alpha
        if b < float("inf"):
            alpha = (a+b)/2
            if alpha == prev_alpha:
                return
            prev_alpha = alpha
        else:
            alpha = 2*a
            prev_alpha = alpha


def armijo(x, d, objective, c1=0.5, alpha=10, rho=0.9):
    nab = objective.grad(device_put(x)).block_until_ready()
    f = objective(x)
    phi_dif0 = np.dot(nab.T, d)
    if phi_dif0 >= 0:
        return 0
    while True:
        phi = objective(x+alpha*d)
        if phi > f + c1*alpha*phi_dif0:
            alpha *= rho
        else:
            return alpha


def check_grad(x, objective):
    n = x.shape[0]
    h = 1e-6
    I = np.eye(n, n)*h
    x_h = I + x
    x_b = -I + x
    grad = np.array(
        [[(objective(x_h[:, i]) - objective(x_b[:, i])) / 2 / h]for i in range(n)])
    grad_ = objective.grad(device_put(x)).block_until_ready()
    diff = np.abs(grad_ - grad)

    assert (diff < 1e-8).all()
