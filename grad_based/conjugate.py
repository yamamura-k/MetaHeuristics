"""
References : 
- https://ja.wikipedia.org/wiki/%E5%85%B1%E5%BD%B9%E5%8B%BE%E9%85%8D%E6%B3%95
- https://ja.wikipedia.org/wiki/%E9%9D%9E%E7%B7%9A%E5%BD%A2%E5%85%B1%E5%BD%B9%E5%8B%BE%E9%85%8D%E6%B3%95
- 基礎数学 IV 最適化理論
"""
import numpy as np


def optimize(dimension, objective):
    try:
        objective.grad(np.zeros(2))
    except AttributeError:
        raise AttributeError(f"{objective.__name__} is not differentiable")
    x = np.random.uniform(*objective.boundaries, size=dimension)
