import numpy as np

import grad_based.conjugate as CG
import grad_based.gradient_descent as GD
import grad_based.nesterov as NV
import grad_based.newton as NW
from benchmarks import log_exp

n, m = 10, 100
func = log_exp(n=n, m=m)
x = np.random.randn(n, 1)
max_iter = 100


def test_CG():
    CG.optimize(x.copy(), func, max_iter)


def test_GD():
    GD.optimize(x.copy(), func, max_iter)


def test_NV():
    NV.optimize(x.copy(), func, max_iter)


def test_NW():
    NW.optimize(x.copy(), func)
