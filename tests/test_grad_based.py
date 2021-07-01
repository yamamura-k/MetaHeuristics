import numpy as np

from benchmarks import log_exp
from grad_based import CG, GD, NV, NW

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
