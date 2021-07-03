import numpy as np

from benchmarks import log_exp
from grad_based import CG, GD, NV, NW

n, m = 10, 100
func = log_exp(n=n, m=m)
x = np.random.randn(n, 1)
max_iter = 100


def test_CG_FR():
    CG.optimize(x.copy(), func, max_iter, method="FR")


def test_CG_PR():
    CG.optimize(x.copy(), func, max_iter, method="PR")


def test_CG_HS():
    CG.optimize(x.copy(), func, max_iter, method="HS")


def test_CG_DY():
    CG.optimize(x.copy(), func, max_iter, method="DY")


def test_CG_heuristic():
    CG.optimize(x.copy(), func, max_iter, method="heuristic")


def test_GD_static():
    GD.optimize(x.copy(), func, max_iter, method="static")


def test_GD_armijo():
    GD.optimize(x.copy(), func, max_iter, method="armijo")


def test_GD_exact():
    GD.optimize(x.copy(), func, max_iter, method="exact")


def test_NV():
    NV.optimize(x.copy(), func, max_iter)


def test_NW():
    NW.optimize(x.copy(), func)
