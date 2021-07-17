from utils.common import FunctionWrapper

import algorithm.grad_based as grad_based
import algorithm.metaheuristics as metaheuristics
import algorithm.nelder_mead as nelder_mead


def optimize(dimension, f, max_iter, algo="CG", maximize=False, *args, **kwargs):
    """

    Parameters
    ----------
    dimension : int
        number of variables
    f : callable
        objective function
    max_iter : int
        maximum number of iteration
    algo : str, optional
        algorithm name, by default "CG". [ABC, BA, GWO, FA, paraABC, paraBA, GD, CG, NV, NW, NM] is valid.
    maximize : bool, optional
        When you want to solve maximization problem, it should be True, by default False
    options : dict, optional
        algorithm specific parameters.
        Coming soon...

    Returns
    -------
    ResultManager
        result of optimization problem

    Raises
    ------
    NotImplementedError
        [description]
    """
    f = FunctionWrapper(f, maximize=maximize, *args, **kwargs)
    if algo == "ABC":
        return metaheuristics.ABC.minimize(dimension, f, max_iter, *args, **kwargs)
    elif algo == "paraABC":
        return metaheuristics.paraABC.minimize(
            dimension, f, max_iter, *args, **kwargs)
    elif algo == "BA":
        return metaheuristics.BA.minimize(dimension, f, max_iter, *args, **kwargs)
    elif algo == "paraBA":
        return metaheuristics.paraBA.minimize(dimension, f, max_iter, *args, **kwargs)
    elif algo == "GWO":
        return metaheuristics.GWO.minimize(dimension, f, max_iter, *args, **kwargs)
    elif algo == "FA":
        return metaheuristics.FA.minimize(dimension, f, max_iter, *args, **kwargs)
    elif algo == "TLBO":
        return metaheuristics.TLBO.minimize(dimension, f, max_iter, *args, **kwargs)
    elif algo == "NM":
        return nelder_mead.minimize(dimension, f, max_iter, *args, **kwargs)
    elif algo == "GD":
        return grad_based.GD.minimize(
            dimension, f, max_iter, *args, **kwargs)
    elif algo == "CG":
        return grad_based.CG.minimize(
            dimension, f, max_iter, *args, **kwargs)
    elif algo == "NV":
        return grad_based.NV.minimize(
            dimension, f, max_iter, *args, **kwargs)
    elif algo == "NW":
        return grad_based.NW.minimize(
            dimension, f, max_iter, *args, **kwargs)
    else:
        raise NotImplementedError
