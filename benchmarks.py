from jax import partial

import jax.numpy as np
from jax import jit

from utils.base import Function, gen_matrix

"""
Reference : https://qiita.com/nabenabe0928/items/08ed6495853c3dd08f1e
"""


class log_exp(Function):
    def __init__(self, A=None, n=10, m=100):
        super().__init__()
        if A is None:
            self.A = gen_matrix(m, n)
        else:
            self.A = A
        self.boundaries = [-.5, .5]

    @partial(jit, static_argnums=(0, ))
    def __call__(self, x):
        return np.log(np.sum(np.exp(self.A@x + 1)))

    def grad(self, x):
        M = np.sum(np.exp(self.A@x + 1))
        _nabla = np.array([sum(a[i]*np.exp(a.T@x + 1)
                          for a in self.A)/M for i in range(len(x))])
        return _nabla

    def hesse(self, x, grad=None):
        if grad is None:
            nabla = self.grad(x)
        else:
            nabla = grad
        _, n = self.A.shape
        H = np.empty((n, n))
        M = np.sum(np.exp(self.A@x + 1))
        for i in range(n):
            for j in range(n):
                H[i][j] = nabla[i]*nabla[j] + \
                    sum(a[i]*a[j]*np.exp(a.T@x + 1)for a in self.A)/M
        return H.reshape((n, n))


class ackley(Function):
    def __init__(self):
        super().__init__()
        self.opt = 0
        self.boundaries = np.array([-32.768, 32.768])

    @partial(jit, static_argnums=(0, ))
    def __call__(self, x):
        t1 = 20
        t2 = - 20 * np.exp(- 0.2 * np.sqrt(1.0 / len(x) * x.T@x))
        t3 = np.e
        t4 = - np.exp(1.0 / len(x) * np.sum(np.cos(2 * np.pi * x)))
        return t1 + t2 + t3 + t4


class sphere(Function):
    def __init__(self):
        super().__init__()
        self.opt = 0
        self.boundaries = np.array([-100, 100])

    @partial(jit, static_argnums=(0, ))
    def __call__(self, x):
        return x.T@x


class rosenbrock(Function):
    def __init__(self):
        super().__init__()
        self.opt = 0
        self.boundaries = np.array([-5, 5])

    @partial(jit, static_argnums=(0, ))
    def __call__(self, x):
        val = np.sum(100*(x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2, axis=0)
        return val


class styblinski(Function):
    def __init__(self, dimension=1):
        super().__init__()
        # approximate optimal value is self.opt * number of variables
        self.opt = -39.166165*dimension
        self.boundaries = np.array([-5, 4])

    @partial(jit, static_argnums=(0, ))
    def __call__(self, x):
        pow_x = x ** 2
        t1 = np.sum(pow_x ** 2)
        t2 = - 16 * np.sum(pow_x)
        t3 = 5 * np.sum(x)
        return 0.5 * (t1 + t2 + t3)


class k_tablet(Function):
    def __init__(self, dimension=1):
        super().__init__()
        self.opt = 0
        self.boundaries = np.array([-5.12, 5.12])
        self.k = int(np.ceil(dimension / 4.0))

    @partial(jit, static_argnums=(0, ))
    def __call__(self, x):
        t1 = x[:self.k].T@x[:self.k]
        t2 = 100 ** 2 * t1
        return t1 + t2


class weighted_sphere(Function):
    def __init__(self, dimension=1):
        super().__init__()
        self.opt = 0
        self.boundaries = np.array([-5.12, 5.12])
        self.coef = np.arange(start=1, step=1, stop=dimension+1)

    @partial(jit, static_argnums=(0, ))
    def __call__(self, x):
        val = self.coef * x * x
        return np.sum(val)


class different_power(Function):
    def __init__(self, dimension=1):
        super().__init__()
        self.opt = 0
        self.boundaries = np.array([-1, 1])
        self.coef = np.arange(start=1, step=1, stop=dimension+1)

    @partial(jit, static_argnums=(0, ))
    def __call__(self, x):
        val = np.sum(np.power(np.abs(x), self.coef))
        return val


class griewank(Function):
    def __init__(self, dimension):
        super().__init__()
        self.opt = 0
        self.boundaries = np.array([-600, 600])
        tmp = np.arange(start=1, step=1, stop=dimension+1)
        self.w = 1.0 / tmp

    @partial(jit, static_argnums=(0, ))
    def __call__(self, x):
        t1 = 1
        t2 = 1.0 / 4000.0 * x.T@x
        t3 = - np.prod(np.cos(x * self.w))
        return t1 + t2 + t3
