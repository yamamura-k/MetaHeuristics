import grad_based.conjugate as CG
from benchmarks import pow


def test_pow():
    dimension = 2
    CG.optimize(dimension, pow())
