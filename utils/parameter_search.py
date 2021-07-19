import os
import pickle as pkl

import algorithm
import optuna


def getParams(algo_name, trial):
    params = dict(
        num_population=trial.suggest_int("num_population", 100, 500, step=50),
        max_iter=trial.suggest_int("max_iter", 20, 500, step=10),
    )
    if algo_name in {"ABC", "paraABC"}:
        params["max_visit"] = trial.suggest_int("max_visit", 1, 20)
    elif algo_name in {"BA", "paraBA"}:
        params["f_min"] = trial.suggest_int("f_min", 0, 50)
        params["f_max"] = trial.suggest_int("f_max", 50, 500, step=10)
        params["seleection_max"] = trial.suggest_int(
            "selection_max", 5, 50, step=5)
        params["alpha"] = trial.suggest_uniform("alpha", 0, 1)
        params["gamma"] = trial.suggest_uniform("gamma", 0, 1)
    elif algo_name == "GWO":
        pass
    elif algo_name == "FA":
        params["alpha"] = trial.suggest_uniform("alpha", 0, 1)
        params["beta"] = trial.suggest_uniform("beta", 0, 10)
        params["gamma"] = trial.suggest_uniform("gamma", 0, 10)
    elif algo_name == "TLBO":
        pass
    elif algo_name == "NM":
        del params["num_population"]
        params["alpha"] = trial.suggest_uniform("alpha", 0, 5)
        params["gamma"] = trial.suggest_uniform("gamma", 0, 5)
        params["rho"] = trial.suggest_uniform("rho", 0, 1)
        params["sigma"] = trial.suggest_uniform("sigma", 0, 1)
    elif algo_name in {"GD", "CG", "NV", "NW"}:
        del params["num_population"]
        lin_search = ["exact", "armijo"]
        if algo_name != "CG":
            params["alpha"] = trial.suggest_uniform("alpha", 0, 1)
            lin_search.append("static")
        params["method"] = trial.suggest_categorical(
            "method", lin_search)
        if algo_name == "CG":
            params["beta_method"] = trial.suggest_categorical(
                "beta_method", ["default", "heuristic", "PR", "FR", "DY", "HS"])
        if algo_name == "NW":
            params["eps"] = trial.suggest_uniform("alpha", 1e-9, 1e-4)
    else:
        raise NotImplementedError
    return params


def getBestParams(dimension, f, algo_name, n_jobs=2, n_trials=100, seed=0, is_search=False, save_dir="./params"):
    def objective(trial, dimension, f, algo_name):
        params = getParams(algo_name, trial)
        return algorithm.optimize(dimension, f, algo=algo_name, EXP=True, **params).best_obj

    os.makedirs(save_dir, exist_ok=True)
    filename = f"best_param_{algo_name}_{f.name}_dimension{dimension}.pkl"
    filepath = os.path.join(save_dir, filename)
    # if parameter file exists and is_search is False, return it
    if os.path.isfile(filepath) and not is_search:
        with open(os.path.join(save_dir, filename), "wb") as f:
            return pkl.load(f)
    # otherwize, search best parameters with optuna
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda trial: objective(
        trial, dimension=dimension, f=f, algo_name=algo_name), n_trials=n_trials, n_jobs=n_jobs)
    best_params = study.best_params

    with open(filepath, "wb") as f:
        pkl.dump(best_params, f, protocol=3)

    return best_params
