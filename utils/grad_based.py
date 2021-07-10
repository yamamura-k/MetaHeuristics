import numpy as np
from scipy.optimize import minimize_scalar


def calc_stepsize(method, x, alpha, objective):
    if method == "wolfe":
        return wolfe(x, objective)
    elif method == "armijo":
        return armijo(x, objective)
    elif method == "static":
        return alpha
    elif method == "exact":
        def phi(alpha):
            return objective(x-alpha*objective.grad(x))
        return np.float(minimize_scalar(phi).x)
    else:
        raise NotImplementedError


def wolfe(x, objective, c1=1e-5, c2=1-1e-5, alpha=10):
    a = 0
    b = np.inf
    nab = objective.grad(x)
    d = -nab
    f = objective(x)
    phi_dif0 = np.dot(nab.T, d)
    assert phi_dif0 < 0
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


def armijo(x, objective, c1=0.5, alpha=10, rho=0.9):
    nab = objective.grad(x)
    d = -nab
    f = objective(x)
    phi_dif0 = np.dot(nab.T, d)
    assert phi_dif0 < 0
    while True:
        phi = objective(x+alpha*d)
        if phi > f + c1*alpha*phi_dif0:
            alpha *= rho
        else:
            return alpha
