import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


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


class ContinuousOptResult(object):
    def __init__(self, objective, algo_name, logger) -> None:
        super().__init__()
        self.objective = objective
        self.algo_name = algo_name
        self.logger = logger

        self.best_obj = np.inf
        self.best_x = None

        self.pos1 = []
        self.pos2 = []
        self.best_pos1 = []
        self.best_pos2 = []

        self.div_max = -1
        self.divs = []
        self.div_maxs = []

    def post_process_per_iter(self, x, best_x, iteration):
        assert len(x.shape) == 2
        _, dimension = x.shape
        if dimension == 2:
            self.pos1.append(x[:, 0].tolist())
            self.pos2.append(x[:, 1].tolist())
            self.best_pos1.append(best_x[0])
            self.best_pos2.append(best_x[1])

        tmp_obj = self.objective(best_x)
        if tmp_obj < self.best_obj:
            self.best_obj = tmp_obj
            self.best_x = best_x.copy()

        div = dimension_wise_diversity_measurement(x)
        self.div_max = max(self.div_max, div)
        self.divs.append(div)
        self.div_maxs.append(self.div_max)

        self.logger.debug(
            f"iteration {iteration} [ best objective ] {self.best_obj}")
        self.logger.debug(f"XPL is {div/self.div_max * 100}")
        self.logger.debug(
            f"XPT is {abs(div - self.div_max)/self.div_max * 100}")

    def plot(self, save_dir="./images"):
        if len(self.pos1) == 0:
            self.logger.warning(
                "No infomation is stored. Cannot create animation.")
            return
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
            plt.plot(self.pos1[i], self.pos2[i], 'o', color="orange",
                     markeredgecolor="black")
            plt.plot(self.best_pos1[i], self.best_pos2[i], 'o',
                     color="red", markeredgecolor="black")
            plt.title('step={}'.format(i))

        ani = animation.FuncAnimation(fig, plot, len(self.pos1), interval=200)
        os.makedirs(save_dir, exist_ok=True)
        ani.save(
            f"{save_dir}/{self.objective.name}_{self.algo_name}.gif", writer="pillow")
        # ani.save(f"{save_dir}/{self.objective.name}_{algo_name}.mp4", writer="ffmpeg")
        plt.clf()
        plt.close()
