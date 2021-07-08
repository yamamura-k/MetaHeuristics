from multiprocessing import Value

import numpy as np


def value_to_numpy(v):
    return np.frombuffer(v.get_obj())


def value_to_numpy2(v):
    return np.array([v[i] for i in range(len(v))])


def numpy_to_value(x, ctype, init_val=None):
    tmp = x.shape
    size = ctype * tmp[-1]
    if len(tmp) > 1:
        size = size * tmp[0]
    if init_val is None:
        v = Value(size)
        value_to_numpy(v)[:] = x
    else:
        v = Value(size, init_val)
    return v