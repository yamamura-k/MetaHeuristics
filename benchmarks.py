import numpy as np
"""
Reference : https://qiita.com/nabenabe0928/items/08ed6495853c3dd08f1e
"""


class ackley():
    def __init__(self):
        self.name = "Ackley"
        self.opt = 0
        self.boundaries = np.array([-32.768, 32.768])

    def __call__(self, x):
        t1 = 20
        t2 = - 20 * np.exp(- 0.2 * np.sqrt(1.0 / len(x) * np.sum(x ** 2)))
        t3 = np.e
        t4 = - np.exp(1.0 / len(x) * np.sum(np.cos(2 * np.pi * x)))
        return t1 + t2 + t3 + t4


class sphere():
    def __init__(self):
        self.name = "Sphere"
        self.opt = 0
        self.boundaries = np.array([-100, 100])

    def __call__(self, x):
        return np.sum(x ** 2)


class rosenbrock():
    def __init__(self):
        self.name = "Rosenbrock"
        self.opt = 0
        self.boundaries = np.array([-5, 5])

    def __call__(self, x):
        val = 0
        for i in range(0, len(x) - 1):
            t1 = 100 * (x[i + 1] - x[i] ** 2) ** 2
            t2 = (x[i] - 1) ** 2
            val += t1 + t2
        return val


class styblinski():
    def __init__(self):
        self.name = "Styblinski-Tang"
        self.opt = -39.166165
        self.boundaries = np.array([-5, 4])

    def __call__(self, x):
        t1 = np.sum(x ** 4)
        t2 = - 16 * np.sum(x ** 2)
        t3 = 5 * np.sum(x)
        return 0.5 * (t1 + t2 + t3)


class k_tablet():
    def __init__(self):
        self.name = "k-tablet"
        self.opt = 0
        self.boundaries = np.array([-5.12, 5.12])

    def __call__(self, x):
        k = int(np.ceil(len(x) / 4.0))
        t1 = np.sum(x[:k] ** 2)
        t2 = 100 ** 2 * np.sum(x[k:] ** 2)
        return t1 + t2


class weighted_sphere():
    def __init__(self):
        self.name = "Weighted Sphere function or hyper ellipsodic function"
        self.opt = 0
        self.boundaries = np.array([-5.12, 5.12])

    def __call__(self, x):
        val = np.array([(i + 1) * xi ** 2 for i, xi in enumerate(x)])
        return np.sum(val)


class different_power():
    def __init__(self):
        self.name = "Sum of different power function"
        self.opt = 0
        self.boundaries = np.array([-1, 1])

    def __call__(self, x):
        val = 0
        for i, v in enumerate(x):
            val += np.abs(v) ** (i + 2)
        return val


class griewank():
    def __init__(self):
        self.name = "Griewank"
        self.opt = 0
        self.boundaries = np.array([-600, 600])

    def __call__(self, x):
        w = np.array([1.0 / np.sqrt(i + 1) for i in range(len(x))])
        t1 = 1
        t2 = 1.0 / 4000.0 * np.sum(x ** 2)
        t3 = - np.prod(np.cos(x * w))
        return t1 + t2 + t3
