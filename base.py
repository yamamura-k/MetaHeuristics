from abc import ABCMeta, abstractmethod

import numpy as np


def gen_matrix(m, n):
    return np.random.randn(m, n)


class Function(metaclass=ABCMeta):
    def __init__(self):
        self.name = None
        self.opt = None
        self.__boundaries = None

    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError

    @property
    def boundaries(self):
        pass

    @boundaries.getter
    def boundaries(self):
        if self.__boundaries is None:
            self.boundaries = (-np.inf, np.inf)
        return self.__boundaries

    @boundaries.setter
    def boundaries(self, bound):
        self.__boundaries = bound

    def grad(self, x):
        """approximate gradient
        """
        n = x.shape[0]
        h = 1e-6
        I = np.eye(n, n)*h
        x_h = I + x
        x_b = -I + x
        _grad = np.array(
            [[(self(x_h[:, i]) - self(x_b[:, i])) / 2 / h]for i in range(n)])
        return _grad

    def hesse(self, x):
        raise NotImplementedError

    def _projection(self, x):
        x = np.asarray(x)
        if (x < self.boundaries[0]).any() or (x > self.boundaries[1]).any():
            x = np.clip(x, self.boundaries[0], self.boundaries[1])
        return x
