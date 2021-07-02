from heapq import heapify, heappop, heappush

import numpy as np

from .utils import randomize


def optimize(dimension, num_population, objective, max_iter, *args, **kwargs):
    x = randomize((num_population, dimension), objective)
    heap = heapify([(objective(x[i]), i) for i in range(num_population)])
    best = [None] * 3
    for i in range(3):
        best[i] = heappop(heap)
    best = np.array(best)

    a = np.zeros(dimension) + 2
    r1 = np.random.random(dimension)
    C = 2*np.random.random(dimension)
    A = a*r1-a

    X_s = x[best[:, 1]].copy()
    A_s = np.broadcast_to(A, (dimension, 3))
    C_s = np.broadcast_to(C, (dimension, 3))

    for _ in range(max_iter):
        D = np.abs(C_s*X_s - x)
        x = np.sum(X_s-A_s*D)/3
        A -= A/max_iter
        C = 2*np.random.random(dimension)
        heap = heapify([(objective(x[i]), i) for i in range(num_population)])
        for i in range(3):
            # 更新方法は要検討
            best[i] = heappop(heap)
            A_s[i] = A.copy()
            C_s[i] = C.copy()
        best = np.array(best)
