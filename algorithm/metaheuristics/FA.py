import numpy as np
from jax import device_put
from utils import ResultManager, getInitialPoint, setup_logger

np.random.seed(0)
logger = setup_logger(__name__)


def minimize(dimension, objective, max_iter, num_population=100, beta=1, gamma=1, alpha=0.2, *args, **kwargs):
    x = getInitialPoint((num_population, dimension), objective)
    I = np.array([objective(device_put(t)).block_until_ready() for t in x])
    best_idx = np.argmin(I)
    best_x = x[best_idx].copy()
    result = ResultManager(objective, __name__, logger, *args, **kwargs)
    result.post_process_per_iter(x, best_x, -1)
    for t in range(max_iter):
        for i in range(num_population):
            # vector implementation
            better_x = x[np.where(I < I[i])].copy()
            better_x -= x[i]
            norm = np.sum(better_x*better_x, axis=0)
            rand = np.random.random(size=(better_x.shape[0], 1))
            rand = np.broadcast_to(rand, better_x.shape).copy()
            x[i] += np.sum(beta*np.exp(-gamma*norm) * better_x +
                           alpha*(rand-0.5), axis=0)
            assert (x[np.where(I < I[i])] != better_x).all()
            I[i] = objective(x[i])
            if I[i] < I[best_idx]:
                best_idx = i
                best_x = x[i].copy()
            continue
            # naive implementation
            for j in range(num_population):
                if I[j] <= I[i]:
                    tmp = x[j] - x[i]
                    x[i] += beta*np.exp(-gamma*tmp.T@tmp) * \
                        tmp + alpha*(np.random.random()-0.5)
                    I[i] = objective(x[i])
                    if I[i] < I[best_idx]:
                        best_idx = i
                        best_x = x[i].copy()
        result.post_process_per_iter(x, best_x, t)
    return result
