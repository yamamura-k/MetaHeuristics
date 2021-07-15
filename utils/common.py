import os
from typing import Callable

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from utils.base import Function


def randomize(shape, objective):
    try:
        return np.random.uniform(*objective.boundaries, size=shape)
    except AttributeError:
        return np.random.random(size=shape)


def dimension_wise_diversity_measurement(x):
    """
    POPULATION DIVERSITY MAINTENANCE IN BRAIN
    STORM OPTIMIZATION ALGORITHM
    Shi Cheng, Yuhui Shi, Quande Qin, Qingyu Zhang and Ruibin Bai
    """
    assert len(x.shape) == 2
    median = np.broadcast_to(np.median(x, axis=0), x.shape).copy()
    div = np.abs(median - x)
    div = np.mean(div)

    return div


class FunctionWrapper(Function):
    def __init__(self, objective, grad=None, hesse=None, lb=None, ub=None, opt=None, name=None, *args, **kwargs):
        super().__init__()
        assert isinstance(objective, Callable)
        assert (grad is None) or isinstance(grad, Callable)
        assert (hesse is None) or isinstance(hesse, Callable)
        self.objective = objective
        self._grad = grad
        self._hesse = hesse
        self.boundaries = (lb, ub)
        if opt is not None:
            self.opt = opt
        if name is not None:
            self.name = name

    def __call__(self, x):
        self._projection(x)
        return self.objective(x)

    def grad(self, x):
        self._projection(x)
        if self._grad is None:
            return super().grad(x)
        else:
            return self._grad(x)

    def hesse(self, x):
        self._projection(x)
        if self._hesse is None:
            return super().hesse(x)
        else:
            return self._hesse(x)


class ResultManager(object):
    def __init__(self, objective, algo_name, logger, limit=np.inf, EXP=False, *args, **kwargs) -> None:
        super().__init__()
        self.objective = objective
        self.algo_name = algo_name
        self.logger = logger

        self.best_obj = np.inf
        self.best_x = None
        self.not_updated = 0
        self.limit = limit
        self.optimal = False

        self.EXP = EXP

        self.pos = []
        self.best_pos = []

        self.div_max = -1
        self.divs = []
        self.div_maxs = []

    def post_process_per_iter(self, x, best_x, iteration):
        assert len(x.shape) == 2
        if not self.EXP:
            self.pos.append(x)
            self.best_pos.append(best_x)

        tmp_obj = np.asscalar(self.objective(best_x))
        if tmp_obj < self.best_obj:
            self.best_obj = tmp_obj
            self.best_x = best_x.copy()
            self.not_updated = 0
        else:
            self.not_updated += 1

        div = dimension_wise_diversity_measurement(x)
        self.div_max = max(self.div_max, div)
        self.divs.append(div)
        self.div_maxs.append(self.div_max)

        self.logger.debug(
            f"iteration {iteration} [ best objective ] {self.best_obj}")
        self.logger.debug(f"XPL is {div/self.div_max * 100}")
        self.logger.debug(
            f"XPT is {abs(div - self.div_max)/self.div_max * 100}")

        if self.not_updated > self.limit:
            self.not_updated = 0
            x = randomize(x.shape, self.objective)
            self.logger.warning(
                "Randomize each population for diversification")

        if np.isclose(self.best_obj, self.objective.opt):
            self.logger.info("Optimal solution is found.")
            self.optimal = True
            return True

        return False

    def plot(self, save_dir="./images"):
        if len(self.pos) == 0:
            self.logger.warning(
                "No infomation is stored. Cannot create animation.")
            return
        self.pos = np.asarray(self.pos)
        self.best_pos = np.asarray(self.best_pos)
        _iter, a, b = self.pos.shape
        if b == 1 and a > 1:
            self.pos = self.pos.reshape((_iter, b, a))

        fig = plt.figure()
        boundaries = list(self.objective.boundaries)
        func_y = []
        diff_num = 50
        diff = boundaries[1] - boundaries[0]
        for x1 in range(diff_num):
            x1 = boundaries[0] + (x1/diff_num)*diff
            d = []
            for x2 in range(diff_num):
                x2 = boundaries[0] + (x2/diff_num)*diff
                y = self.objective(np.asarray([x2, x1]))
                d.append(y)
            func_y.insert(0, d)

        extent = tuple(boundaries*2)
        plt.imshow(func_y, interpolation="nearest",  cmap="jet", extent=extent)
        plt.colorbar()

        def plot(i):
            plt.cla()
            plt.imshow(func_y, interpolation="nearest",
                       cmap="jet", extent=extent)
            plt.plot(self.pos[i][:, 0], self.pos[i][:, 1], 'o', color="orange",
                     markeredgecolor="black")
            plt.plot(self.best_pos[i][0], self.best_pos[i][1], 'o',
                     color="red", markeredgecolor="black")
            plt.title('step={}'.format(i))

        ani = animation.FuncAnimation(fig, plot, len(self.pos), interval=200)
        os.makedirs(save_dir, exist_ok=True)
        ani.save(
            f"{save_dir}/{self.objective.name}_{self.algo_name}.gif", writer="pillow")
        # ani.save(f"{save_dir}/{self.objective.name}_{algo_name}.mp4", writer="ffmpeg")
        plt.clf()
        plt.close()
