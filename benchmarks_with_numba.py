import numpy as np
from numba import types
from numba.experimental import jitclass

from utils import gen_matrix, projection

"""
Reference : https://qiita.com/nabenabe0928/items/08ed6495853c3dd08f1e
"""

INF = 1 << 63
spec = [
    ("name", types.string),
    ("opt", types.double),
    ("boundaries", types.double[:])
]

spec1 = [
    ("name", types.string),
    ("A", types.double[:])
]


@jitclass(spec1)
class log_exp:
    def __init__(self, A=None, n=10, m=100):
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

    @property
    def boundaries(self):
        pass

    @boundaries.getter
    def boundaries(self):
        if self.__boundaries is None:
            return [-INF, INF]
        else:
            return self.__boundaries

    @boundaries.setter
    def boundaries(self, bound):
        self.__boundaries = bound


@jitclass(spec)
class ackley:
    def __init__(self):
        self.name = "Ackley"
        self.opt = 0
        self.boundaries = np.array([-32.768, 32.768])

    def __call__(self, x):
        projection(x, self.boundaries)
        t1 = 20
        t2 = - 20 * np.exp(- 0.2 * np.sqrt(1.0 / len(x) * np.sum(x ** 2)))
        t3 = np.e
        t4 = - np.exp(1.0 / len(x) * np.sum(np.cos(2 * np.pi * x)))
        return t1 + t2 + t3 + t4

    @property
    def boundaries(self):
        pass

    @boundaries.getter
    def boundaries(self):
        if self.__boundaries is None:
            return [-INF, INF]
        else:
            return self.__boundaries

    @boundaries.setter
    def boundaries(self, bound):
        self.__boundaries = bound

    def grad(self, x):
        raise NotImplementedError

    def hesse(self, x):
        raise NotImplementedError


@jitclass(spec)
class sphere:
    def __init__(self):
        self.name = "Sphere"
        self.opt = 0
        self.boundaries = np.array([-100, 100])

    def __call__(self, x):
        projection(x, self.boundaries)
        return np.sum(x ** 2)

    @property
    def boundaries(self):
        pass

    @boundaries.getter
    def boundaries(self):
        if self.__boundaries is None:
            return [-INF, INF]
        else:
            return self.__boundaries

    @boundaries.setter
    def boundaries(self, bound):
        self.__boundaries = bound

    def grad(self, x):
        raise NotImplementedError

    def hesse(self, x):
        raise NotImplementedError


@jitclass(spec)
class rosenbrock:
    def __init__(self):
        self.name = "Rosenbrock"
        self.opt = 0
        self.boundaries = np.array([-5, 5])

    def __call__(self, x):
        projection(x, self.boundaries)
        val = 0
        for i in range(0, len(x) - 1):
            t1 = 100 * (x[i + 1] - x[i] ** 2) ** 2
            t2 = (x[i] - 1) ** 2
            val += t1 + t2
        return val

    @property
    def boundaries(self):
        pass

    @boundaries.getter
    def boundaries(self):
        if self.__boundaries is None:
            return [-INF, INF]
        else:
            return self.__boundaries

    @boundaries.setter
    def boundaries(self, bound):
        self.__boundaries = bound

    def grad(self, x):
        raise NotImplementedError

    def hesse(self, x):
        raise NotImplementedError


@jitclass(spec)
class styblinski:
    def __init__(self, dimension=1):
        self.name = "Styblinski-Tang"
        # approximate optimal value is self.opt * number of variables
        self.opt = -39.166165*dimension
        self.boundaries = np.array([-5, 4])

    def __call__(self, x):
        projection(x, self.boundaries)
        t1 = np.sum(x ** 4)
        t2 = - 16 * np.sum(x ** 2)
        t3 = 5 * np.sum(x)
        return 0.5 * (t1 + t2 + t3)

    @property
    def boundaries(self):
        pass

    @boundaries.getter
    def boundaries(self):
        if self.__boundaries is None:
            return [-INF, INF]
        else:
            return self.__boundaries

    @boundaries.setter
    def boundaries(self, bound):
        self.__boundaries = bound

    def grad(self, x):
        raise NotImplementedError

    def hesse(self, x):
        raise NotImplementedError


@jitclass(spec)
class k_tablet:
    def __init__(self):

        self.name = "k-tablet"
        self.opt = 0
        self.boundaries = np.array([-5.12, 5.12])

    def __call__(self, x):
        projection(x, self.boundaries)
        k = int(np.ceil(len(x) / 4.0))
        t1 = np.sum(x[:k] ** 2)
        t2 = 100 ** 2 * np.sum(x[k:] ** 2)
        return t1 + t2

    @property
    def boundaries(self):
        pass

    @boundaries.getter
    def boundaries(self):
        if self.__boundaries is None:
            return [-INF, INF]
        else:
            return self.__boundaries

    @boundaries.setter
    def boundaries(self, bound):
        self.__boundaries = bound

    def grad(self, x):
        raise NotImplementedError

    def hesse(self, x):
        raise NotImplementedError


@jitclass(spec)
class weighted_sphere:
    def __init__(self):

        self.name = "Weighted Sphere function or hyper ellipsodic function"
        self.opt = 0
        self.boundaries = np.array([-5.12, 5.12])

    def __call__(self, x):
        projection(x, self.boundaries)
        val = np.array([(i + 1) * xi ** 2 for i, xi in enumerate(x)])
        return np.sum(val)

    @property
    def boundaries(self):
        pass

    @boundaries.getter
    def boundaries(self):
        if self.__boundaries is None:
            return [-INF, INF]
        else:
            return self.__boundaries

    @boundaries.setter
    def boundaries(self, bound):
        self.__boundaries = bound

    def grad(self, x):
        raise NotImplementedError

    def hesse(self, x):
        raise NotImplementedError


@jitclass(spec)
class different_power:
    def __init__(self):

        self.name = "Sum of different power function"
        self.opt = 0
        self.boundaries = np.array([-1, 1])

    def __call__(self, x):
        projection(x, self.boundaries)
        val = 0
        for i, v in enumerate(x):
            val += np.abs(v) ** (i + 2)
        return val

    @property
    def boundaries(self):
        pass

    @boundaries.getter
    def boundaries(self):
        if self.__boundaries is None:
            return [-INF, INF]
        else:
            return self.__boundaries

    @boundaries.setter
    def boundaries(self, bound):
        self.__boundaries = bound

    def grad(self, x):
        raise NotImplementedError

    def hesse(self, x):
        raise NotImplementedError


@jitclass(spec)
class griewank:
    def __init__(self):

        self.name = "Griewank"
        self.opt = 0
        self.boundaries = np.array([-600, 600])

    def __call__(self, x):
        projection(x, self.boundaries)
        w = np.array([1.0 / np.sqrt(i + 1) for i in range(len(x))])
        t1 = 1
        t2 = 1.0 / 4000.0 * np.sum(x ** 2)
        t3 = - np.prod(np.cos(x * w))
        return t1 + t2 + t3

    @property
    def boundaries(self):
        pass

    @boundaries.getter
    def boundaries(self):
        if self.__boundaries is None:
            return [-INF, INF]
        else:
            return self.__boundaries

    @boundaries.setter
    def boundaries(self, bound):
        self.__boundaries = bound

    def grad(self, x):
        raise NotImplementedError

    def hesse(self, x):
        raise NotImplementedError
