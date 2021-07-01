import numpy as np

from base import Function
from utils import gen_matrix

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
        self.name = "log(sum(a_i x))"

    def __call__(self, x):
        return np.float(np.log(sum(np.exp(a.T@x + 1) for a in self.A)))

    def grad(self, x):
        M = sum(np.exp(a@x + 1) for a in self.A)
        _nabla = np.array([sum(a[i]*np.exp(a.T@x + 1)
                          for a in self.A)/M for i in range(len(x))])
        return _nabla.reshape(len(_nabla), 1)

    def hesse(self, x, grad=None):
        if grad is None:
            nabla = self.grad(x)
        else:
            nabla = grad
        _, n = self.A.shape
        H = np.zeros((n, n))
        M = sum(np.exp(a.T@x + 1) for a in self.A)
        for i in range(n):
            for j in range(n):
                H[i][j] = nabla[i]*nabla[j] + \
                    sum(a[i]*a[j]*np.exp(a.T@x + 1)for a in self.A)/M
        return H.reshape((n, n))


class ackley(Function):
    def __init__(self):
        super().__init__()
        self.name = "Ackley"
        self.opt = 0
        self.boundaries = np.array([-32.768, 32.768])

    def __call__(self, x):
        self._projection(x)
        t1 = 20
        t2 = - 20 * np.exp(- 0.2 * np.sqrt(1.0 / len(x) * np.sum(x ** 2)))
        t3 = np.e
        t4 = - np.exp(1.0 / len(x) * np.sum(np.cos(2 * np.pi * x)))
        return t1 + t2 + t3 + t4


class sphere(Function):
    def __init__(self):
        super().__init__()
        self.name = "Sphere"
        self.opt = 0
        self.boundaries = np.array([-100, 100])

    def __call__(self, x):
        self._projection(x)
        return np.sum(x ** 2)


class rosenbrock(Function):
    def __init__(self):
        super().__init__()
        self.name = "Rosenbrock"
        self.opt = 0
        self.boundaries = np.array([-5, 5])

    def __call__(self, x):
        self._projection(x)
        val = 0
        for i in range(0, len(x) - 1):
            t1 = 100 * (x[i + 1] - x[i] ** 2) ** 2
            t2 = (x[i] - 1) ** 2
            val += t1 + t2
        return val


class styblinski(Function):
    def __init__(self, dimension=1):
        super().__init__()
        self.name = "Styblinski-Tang"
        # approximate optimal value is self.opt * number of variables
        self.opt = -39.166165*dimension
        self.boundaries = np.array([-5, 4])

    def __call__(self, x):
        self._projection(x)
        t1 = np.sum(x ** 4)
        t2 = - 16 * np.sum(x ** 2)
        t3 = 5 * np.sum(x)
        return 0.5 * (t1 + t2 + t3)


class k_tablet(Function):
    def __init__(self):
        super().__init__()
        self.name = "k-tablet"
        self.opt = 0
        self.boundaries = np.array([-5.12, 5.12])

    def __call__(self, x):
        self._projection(x)
        k = int(np.ceil(len(x) / 4.0))
        t1 = np.sum(x[:k] ** 2)
        t2 = 100 ** 2 * np.sum(x[k:] ** 2)
        return t1 + t2


class weighted_sphere(Function):
    def __init__(self):
        super().__init__()
        self.name = "Weighted Sphere function or hyper ellipsodic function"
        self.opt = 0
        self.boundaries = np.array([-5.12, 5.12])

    def __call__(self, x):
        self._projection(x)
        val = np.array([(i + 1) * xi ** 2 for i, xi in enumerate(x)])
        return np.sum(val)


class different_power(Function):
    def __init__(self):
        super().__init__()
        self.name = "Sum of different power function"
        self.opt = 0
        self.boundaries = np.array([-1, 1])

    def __call__(self, x):
        self._projection(x)
        val = 0
        for i, v in enumerate(x):
            val += np.abs(v) ** (i + 2)
        return val


class griewank(Function):
    def __init__(self):
        super().__init__()
        self.name = "Griewank"
        self.opt = 0
        self.boundaries = np.array([-600, 600])

    def __call__(self, x):
        self._projection(x)
        w = np.array([1.0 / np.sqrt(i + 1) for i in range(len(x))])
        t1 = 1
        t2 = 1.0 / 4000.0 * np.sum(x ** 2)
        t3 = - np.prod(np.cos(x * w))
        return t1 + t2 + t3
