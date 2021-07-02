import numpy as np

from .utils import randomize


def optimize(dimension, num_population, objective, max_iter, top_k=3, *args, **kwargs):
    x = randomize((num_population, dimension), objective)
    lis = list(range(num_population))
    obj_vals = [objective(t) for t in x]
    lis.sort(key=lambda x: obj_vals[x])
    best_x = np.zeros((top_k, dimension))
    best_obj = [None] * top_k
    for i in range(top_k):
        best_x[i] = x[lis[i]]
        best_obj[i] = obj_vals[lis[i]]
    ret_obj = best_obj[0]
    ret_x = best_x[0].copy()
    a = np.zeros(dimension) + 2
    r1 = np.random.random(dimension)
    C = 2*np.random.random(dimension)
    A = 2*a*r1-a

    X_s = best_x.copy()
    A_s = np.broadcast_to(A, X_s.shape).copy()
    C_s = np.broadcast_to(C, X_s.shape).copy()

    pos1 = []
    pos2 = []
    best_pos1 = []
    best_pos2 = []

    for _ in range(max_iter):
        prod = C_s*X_s
        tmp = np.array([A_s[i]*np.abs(prod[i] - x)for i in range(top_k)])
        x = np.sum(np.array([X_s[i]-tmp[i]
                   for i in range(top_k)]), axis=0)/top_k
        a -= a/max_iter
        r1 = np.random.random(dimension)
        A = 2*a*r1-a
        C = 2*np.random.random(dimension)
        obj_vals = [objective(t) for t in x]
        lis.sort(key=lambda x: obj_vals[x])

        for i in range(3):
            tmp_o = obj_vals[lis[i]]
            tmp_x = x[lis[i]].copy()
            best_obj[i] = tmp_o
            best_x[i] = tmp_x
            A_s[i] = np.broadcast_to(A, (dimension,)).copy()
            C_s[i] = np.broadcast_to(C, (dimension,)).copy()

        if ret_obj > best_obj[0]:
            ret_obj = best_obj[0]
            ret_x = best_x[0].copy()

        pos1.append(x[:, 0].tolist())
        pos2.append(x[:, 1].tolist())
        best_pos1.append(best_x[1][0])
        best_pos2.append(best_x[1][1])

    return ret_x, ret_obj, (pos1, pos2, best_pos1, best_pos2)
