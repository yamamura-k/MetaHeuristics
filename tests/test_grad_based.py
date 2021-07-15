import numpy as np

from benchmarks import log_exp
from grad_based import CG, GD, NV, NW

n, m = 10, 100
func = log_exp(n=n, m=m)
options = dict(grad=func.grad, hesse=func.hesse)
max_iter = 10


def test_CG_FR():
    CG.optimize(n, func, max_iter, **options, beta_method="FR")


def test_CG_PR():
    CG.optimize(n, func, max_iter, **options, beta_method="PR")


def test_CG_HS():
    CG.optimize(n, func, max_iter, **options, beta_method="HS")


def test_CG_DY():
    CG.optimize(n, func, max_iter, **options, beta_method="DY")


def test_CG_heuristic():
    CG.optimize(n, func, max_iter, **options, beta_method="heuristic")


def test_CG_default():
    CG.optimize(n, func, max_iter, **options, beta_method="default")


def test_GD_static():
    GD.optimize(n, func, max_iter, **options, method="static")


def test_GD_armijo():
    GD.optimize(n, func, max_iter, **options, method="armijo")


def test_GD_exact():
    GD.optimize(n, func, max_iter, **options, method="exact")


def test_NV():
    NV.optimize(n, func, max_iter, **options)


def test_NW():
    NW.optimize(n, func, **options)
