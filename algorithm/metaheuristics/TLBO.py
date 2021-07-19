import numpy as np
from utils import ResultManager, getInitialPoint, setup_logger

np.random.seed(0)
logger = setup_logger(__name__)


def minimize(dimension, objective, max_iter, num_population=100, *args, **kwargs):
    x = getInitialPoint((num_population, dimension), objective)
    obj_vals = np.array([objective(t) for t in x])
    all_idx = np.arange(num_population)
    result = ResultManager(objective, __name__, logger, *args, **kwargs)

    for t in range(max_iter):
        teacher = np.argmin(obj_vals)
        mean = np.mean(x, axis=0)
        Tf = np.round(1+np.random.random())
        r = np.random.random()
        difference_mean = r*(x[teacher] - mean*Tf)
        x_new = x + difference_mean
        comp_idxs = np.random.choice(all_idx, size=num_population)

        tmp = comp_idxs[np.where(comp_idxs == all_idx)].view()
        tmp = (tmp+1) % num_population

        better_idx = np.where(obj_vals < obj_vals[comp_idxs])
        other_idx = np.where(obj_vals >= obj_vals[comp_idxs])
        x_new[better_idx] = x[better_idx] + r * \
            (x[better_idx] - x[comp_idxs[better_idx]])
        x_new[other_idx] = x[other_idx] + r * \
            (x[other_idx] - x[comp_idxs[other_idx]])

        obj_new = np.array([objective(t) for t in x_new])
        update_idxs = np.where(obj_new < obj_vals)

        x[update_idxs] = x_new[update_idxs]
        obj_vals[update_idxs] = obj_new[update_idxs]

        result.post_process_per_iter(x, x[teacher], t)

    return result
