import numpy as np

from benchmarks import log_exp
from grad_based import CG, GD, NV, NW

n, m = 10, 100
func = log_exp(n=n, m=m)
options = dict(grad=func.grad, hesse=func.hesse)
max_iter = 10


def test_CG_FR():
    CG.minimize(n, func, max_iter, **options, beta_method="FR")


def test_CG_PR():
    CG.minimize(n, func, max_iter, **options, beta_method="PR")


def test_CG_HS():
    CG.minimize(n, func, max_iter, **options, beta_method="HS")


def test_CG_DY():
    CG.minimize(n, func, max_iter, **options, beta_method="DY")


def test_CG_heuristic():
    CG.minimize(n, func, max_iter, **options, beta_method="heuristic")


def test_CG_default():
    CG.minimize(n, func, max_iter, **options, beta_method="default")


def test_GD_static():
    GD.minimize(n, func, max_iter, **options, method="static")


def test_GD_armijo():
    GD.minimize(n, func, max_iter, **options, method="armijo")


def test_GD_exact():
    GD.minimize(n, func, max_iter, **options, method="exact")


def test_NV():
    NV.minimize(n, func, max_iter, **options)


def test_NW():
    NW.minimize(n, func, **options)
