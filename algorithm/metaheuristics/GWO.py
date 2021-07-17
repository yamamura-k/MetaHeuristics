import numpy as np
from utils import getInitialPoint, setup_logger
from utils.common import ResultManager

np.random.seed(0)
logger = setup_logger(__name__)


def minimize(dimension, objective, max_iter, num_population=100, top_k=3, *args, **kwargs):
    x = getInitialPoint((num_population, dimension), objective)
    obj_vals = np.array([objective(t) for t in x])
    lis = np.argsort(obj_vals)
    best_x = np.empty((top_k, dimension))
    best_obj = np.empty(top_k)
    for i in range(top_k):
        best_x[i] = x[lis[i]]
        best_obj[i] = obj_vals[lis[i]]

    a = np.full(dimension, 2.0)
    r1 = np.random.random(dimension)
    C = 2*np.random.random(dimension)
    A = 2*a*r1-a

    X_s = best_x.copy()
    A_s = np.broadcast_to(A, X_s.shape).copy()
    C_s = np.broadcast_to(C, X_s.shape).copy()

    result = ResultManager(objective, __name__, logger, *args, **kwargs)
    result.post_process_per_iter(x, best_x[0], -1)

    for t in range(max_iter):
        prod = C_s*X_s
        tmp = np.array([A_s[i]*np.abs(prod[i] - x)for i in range(top_k)])
        x = np.sum(np.array([X_s[i]-tmp[i]
                   for i in range(top_k)]), axis=0)/top_k
        a -= a/max_iter
        r1 = np.random.random(dimension)
        A = 2*a*r1-a
        C = 2*np.random.random(dimension)
        obj_vals = np.array([objective(t) for t in x])
        lis = np.argsort(obj_vals)

        for i in range(3):
            tmp_o = obj_vals[lis[i]]
            tmp_x = x[lis[i]].copy()
            best_obj[i] = tmp_o
            best_x[i] = tmp_x
            A_s[i] = np.broadcast_to(A, (dimension,)).copy()
            C_s[i] = np.broadcast_to(C, (dimension,)).copy()

        if result.post_process_per_iter(x, best_x[0], t):
            break

    return result
