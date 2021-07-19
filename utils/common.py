import os
from typing import Callable

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from utils.base import Function

np.random.seed(0)


def getInitialPoint(shape, objective, initialize_method="random", *args, **kwargs):
    if initialize_method == "random":
        try:
            return np.random.uniform(*objective.boundaries, size=shape)
        except AttributeError:
            return np.random.random(size=shape)
    else:
        return np.zeros(shape=shape)


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


def update_params(base_param: dict, additional: dict):
    """overwrite base parameter dictionary

    Parameters
    ----------
    base_param : dict
        base param dictionary
    additional : dict
        additional param dictionary

    Returns
    -------
    dict
        updated parameter dictionary
    """
    for key in additional:
        base_param[key] = additional[key]
    return base_param


class FunctionWrapper(Function):
    def __init__(self, objective, grad=None, hesse=None, lb=None, ub=None, opt=None, name=None, maximize=False, *args, **kwargs):
        super().__init__()
        assert isinstance(objective, Callable)
        assert (grad is None) or isinstance(grad, Callable)
        assert (hesse is None) or isinstance(hesse, Callable)
        self.objective = objective
        self._grad = grad
        self._hesse = hesse
        self.sign = -1 if maximize else 1
        if isinstance(objective, Function):
            self.boundaries = objective.boundaries
            self.opt = objective.opt
            self.name = objective.name
        if (ub is not None) and (lb is not None):
            self.boundaries = (lb, ub)
        if opt is not None:
            self.opt = opt
        if name is not None:
            self.name = name

    def __call__(self, x):
        self._projection(x)
        return self.sign*self.objective(x)

    def grad(self, x):
        self._projection(x)
        if self._grad is None:
            return self.sign*super().grad(x)
        else:
            return self.sign*self._grad(x)

    def hesse(self, x):
        self._projection(x)
        if self._hesse is None:
            return self.sign*super().hesse(x)
        else:
            return self.sign*self._hesse(x)


class ResultManager(object):
    def __init__(self, objective, algo_name, logger, limit=np.inf, EXP=False, *args, **kwargs) -> None:
        super().__init__()
        self.objective = objective
        self.algo_name = algo_name
        self.logger = logger

        self.best_obj = np.inf
        self.best_x = None
        self.not_updated = 0
        self.num_restart = 0
        limit = min(10, limit) if "grad" in algo_name else limit
        self.limit = limit
        self.optimal = False

        self.EXP = EXP

        self.pos = []
        self.best_pos = []

        self.div_max = -1
        self.divs = []
        self.div_maxs = []

    def post_process_per_iter(self, x, best_x, iteration, beta=None, alpha=None):
        assert len(x.shape) == 2
        x = np.clip(x, *self.objective.boundaries)
        best_x = np.clip(best_x, *self.objective.boundaries)

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
        message = [
            f"iteration {iteration}",
            f"[ best objective ] {self.best_obj}",
            f"({self.objective.name})"]

        if beta is not None:
            message.append(f"[ beta ] {beta}")
        if alpha is not None:
            message.append(f"[ alpha ] {alpha}")
        self.logger.info(" ".join(message))
        self.logger.info(f"XPL is {div/self.div_max * 100}")
        self.logger.info(
            f"XPT is {abs(div - self.div_max)/self.div_max * 100}")
        if alpha is not None and alpha == 0:
            self.num_restart += 1
            x = getInitialPoint(x.shape, self.objective)
            self.logger.warning(
                "getInitialPoint current vector because alpha = 0.")

        if self.not_updated > self.limit:
            self.num_restart += 1
            self.not_updated = 0
            x = getInitialPoint(x.shape, self.objective)
            self.logger.warning(
                "getInitialPoint each population for diversification.")

        if self.best_obj == self.objective.opt:
            self.logger.info("Optimal solution is found.")
            self.optimal = True
        if not self.EXP:
            gtub = np.sum(x > self.objective.boundaries[1])
            ltlb = np.sum(x < self.objective.boundaries[0])
            out_of_bounds = ltlb + gtub
            if out_of_bounds:
                self.logger.critical(
                    f"{out_of_bounds} elements are out of bounds.")
            self.pos.append(x.copy())
            self.best_pos.append(best_x.copy())

        return self.optimal

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

        ani = animation.FuncAnimation(fig, plot, _iter, interval=200)
        os.makedirs(save_dir, exist_ok=True)
        ani.save(
            f"{save_dir}/{self.objective.name}_{self.algo_name}.gif", writer="pillow")
        # ani.save(f"{save_dir}/{self.objective.name}_{algo_name}.mp4", writer="ffmpeg")
        plt.clf()
        plt.close()
