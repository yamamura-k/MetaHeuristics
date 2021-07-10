import os
from abc import ABCMeta, abstractmethod

import matplotlib.animation as animation
import matplotlib.pyplot as plt
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
        raise NotImplementedError

    def hesse(self, x):
        raise NotImplementedError

    def _projection(self, x):
        x = np.asarray(x)
        if (x < self.boundaries[0]).any() or (x > self.boundaries[1]).any():
            x = np.clip(x, self.boundaries[0], self.boundaries[1])
        return x

    def plot(self, pos1, pos2, best_pos1, best_pos2,
             save_dir="./images", algo_name="tmp"):
        fig = plt.figure()
        boundaries = list(self.boundaries)
        func_y = []
        diff_num = 50
        diff = boundaries[1] - boundaries[0]
        for x1 in range(diff_num):
            x1 = boundaries[0] + (x1/diff_num)*diff
            d = []
            for x2 in range(diff_num):
                x2 = boundaries[0] + (x2/diff_num)*diff
                y = self(np.asarray([x2, x1]))
                d.append(y)
            func_y.insert(0, d)

        extent = tuple(boundaries*2)
        plt.imshow(func_y, interpolation="nearest",  cmap="jet", extent=extent)
        plt.colorbar()

        def plot(i):
            plt.cla()
            plt.imshow(func_y, interpolation="nearest",
                       cmap="jet", extent=extent)
            plt.plot(pos1[i], pos2[i], 'o', color="orange",
                     markeredgecolor="black")
            plt.plot(best_pos1[i], best_pos2[i], 'o',
                     color="red", markeredgecolor="black")
            plt.title('step={}'.format(i))

        ani = animation.FuncAnimation(fig, plot, len(pos1), interval=200)
        os.makedirs(save_dir, exist_ok=True)
        ani.save(f"{save_dir}/{self.name}_{algo_name}.gif", writer="pillow")
        # ani.save(f"{save_dir}/{self.name}_{algo_name}.mp4", writer="ffmpeg")
        plt.clf()
        plt.close()
