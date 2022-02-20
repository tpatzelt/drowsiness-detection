from sklearnex import patch_sklearn

patch_sklearn()

import pickle
from pathlib import Path

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

from drowsiness_detection import config
from drowsiness_detection.data import get_train_test_splits

ex = Experiment("grid_search_kss")
ex.observers.append(FileStorageObserver(Path(__file__).parent.parent.joinpath("logs")))


@ex.config
def base():
    data_id = "10_sec"
    cross_val_params = {
        "n_splits": 4,
        "n_repeats": 1
    }
    grid_search_params = {
        "scoring": "roc_auc",
        "n_jobs": -1,
        "error_score": 0,
        "verbose": 2
    }
    model_params = {
        "name": None,
        "param_grid": {}
    }


@ex.named_config
def logistic_regression():
    model_params = {
        "name": "LogisticRegression",
        "param_grid": {
            "max_iter": [1000, 2500],
            "C": [.01, .1, 1, 10, 100],
            "solver": ["newton-cg", "liblinear"],
            "penalty": ["l2"]
        }
    }


@ex.named_config
def random_forest():
    model_params = {
        "name": "RandomForestClassifier",
        "param_grid": {
            'n_estimators': [1000,1500, 2000],
            'max_depth': [10, 50, 100],
            'min_samples_leaf': [1, 10],
            "max_features": ['sqrt', 'log2'],
            'criterion': ['gini', 'entropy']}
    }


def parse_data_id(data_id: str):
    if data_id == "10_sec":
        X_train, y_train, _, _ = get_train_test_splits(config.TEN_SEC_TRAIN_TEST_SPLIT_PATH)
    elif data_id == "20_sec":
        X_train, y_train, _, _ = get_train_test_splits(config.TWENTY_SEC_TRAIN_TEST_SPLIT_PATH)
    elif data_id == "60_sec":
        X_train, y_train, _, _ = get_train_test_splits(config.SIXTY_SEC_TRAIN_TEST_SPLIT_PATH)
    else:
        raise ValueError
    return X_train, y_train


def parse_model_name(model_name: str):
    if model_name == "RandomForestClassifier":
        model = RandomForestClassifier()
    elif model_name == "LogisticRegression":
        model = LogisticRegression()
    else:
        raise ValueError
    return model


@ex.automain
def run(data_id: str, cross_val_params: dict, grid_search_params: dict, model_params: dict):
    X, y = parse_data_id(data_id=data_id)
    model = parse_model_name(model_name=model_params["name"])

    cv = RepeatedStratifiedKFold(**cross_val_params)
    grid_search = GridSearchCV(
        estimator=model, param_grid=model_params["param_grid"], cv=cv, **grid_search_params)

    grid_result = grid_search.fit(X, y)

    ex.info[grid_result.scoring] = float(grid_result.best_score_)
    ex.info["best_params"] = grid_result.best_params_

    result_path = Path("grid_result.pkl")
    with open(result_path, "wb") as fp:
        pickle.dump(file=fp, obj=grid_result)
    ex.add_artifact(result_path, name="grid_search_result.pkl")
    result_path.unlink()
