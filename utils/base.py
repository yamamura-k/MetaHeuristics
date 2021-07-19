from abc import ABCMeta, abstractmethod
from functools import partial

import numpy as np
from jax import jit

from utils.logging import setup_logger


def gen_matrix(m, n):
    np.random.seed(0)
    return np.random.randn(m, n)


logger = setup_logger(__name__)


class Function(metaclass=ABCMeta):
    def __init__(self):
        self.__name = self.__class__.__name__
        self.opt = -np.inf
        self.__boundaries = None

    @abstractmethod
    @partial(jit, static_argnums=0)
    def __call__(self, x):
        raise NotImplementedError

    @property
    def boundaries(self):
        pass

    @property
    def name(self):
        pass

    @name.getter
    def name(self):
        return self.__name

    @name.setter
    def name(self, _name):
        self.__name = _name

    @boundaries.getter
    def boundaries(self):
        if self.__boundaries is None:
            bound = 1e9
            self.boundaries = (-bound, bound)
        return self.__boundaries

    @boundaries.setter
    def boundaries(self, bound):
        self.__boundaries = bound

    @partial(jit, static_argnums=0)
    def grad(self, x):
        """approximate gradient
        """
        logger.warning("Use approximate gradient.")

        n = x.shape[0]
        h = 1e-6
        I = np.eye(n, n)*h
        x_h = I + x
        x_b = -I + x
        _grad = np.array(
            [[(self(x_h[:, i]) - self(x_b[:, i])) / 2 / h]for i in range(n)])
        return _grad

    @partial(jit, static_argnums=0)
    def hesse(self, x):
        raise NotImplementedError
