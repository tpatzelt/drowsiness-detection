from sklearnex import patch_sklearn

patch_sklearn()
from tempfile import mkdtemp
from shutil import rmtree
import pickle
import ConfigSpace.hyperparameters as CSH

from hpbandster_sklearn import HpBandSterSearchCV
import ConfigSpace as CS
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

from drowsiness_detection import config
from drowsiness_detection.data import get_train_test_splits, get_identifier_array_train_test_split, \
    drop_by_identifier, session_type_mapping

ex = Experiment("grid_search_kss")
ex.observers.append(FileStorageObserver(Path(__file__).parent.parent.joinpath("logs")))

param_distribution = CS.ConfigurationSpace()


@ex.config
def base():
    recording_frequency = 30
    window_in_sec = 10
    cross_val_params = {
        "n_splits": 4,
        "n_repeats": 1
    }
    grid_search_params = {
        "n_iter": None,
        "max_budget": None,
        "optimizer": "bohb",
        "scoring": "roc_auc",
        "n_jobs": -1,
        "error_score": 0,
        "verbose": 1
    }
    model_params = {
        "name": None,
        "param_grid": {}
    }
    exclude_by = "a"


@ex.named_config
def logistic_regression():
    grid_search_params = {
        "n_iter": 10,
        "max_budget": 100,
        "resource_name": "max_iters",
        "resource_type": int
    }

    param_distribution.add_hyperparameter(
        CSH.UniformFloatHyperparameter("C", lower=.001, upper=100, log=True))
    param_distribution.add_hyperparameter(
        CSH.CategoricalHyperparameter("solver", choices=["newton-cg", "liblinear"]))
    param_distribution.add_hyperparameter(CSH.CategoricalHyperparameter("penalty", choices=["l2"]))
    model_params = {
        "name": "LogisticRegression",
        "param_distribution": param_distribution

        #     {
        #     "C": [.01, .1, 1, 10, 100],
        #     "solver": ["newton-cg", "liblinear"],
        #     "penalty": ["l2"]
        # }
    }


@ex.named_config
def random_forest():
    model_params = {
        "name": "RandomForestClassifier",
        "param_grid": {
            'max_depth': [10, 50, 100],
            'min_samples_leaf': [1, 10],
            "max_features": ['sqrt', 'log2'],
            'criterion': ['gini', 'entropy']}
    }


def parse_model_name(model_name: str):
    if model_name == "RandomForestClassifier":
        model = RandomForestClassifier()
    elif model_name == "LogisticRegression":
        model = LogisticRegression()
    else:
        raise ValueError
    return model


@ex.automain
def run(recording_frequency: int, window_in_sec: int, cross_val_params: dict,
        grid_search_params: dict, model_params: dict, exclude_by: str):
    config.set_paths(frequency=recording_frequency, seconds=window_in_sec)
    cache_dir = mkdtemp()

    X_train, y_train, _, _ = get_train_test_splits(config.PATHS.TRAIN_TEST_SPLIT)
    train_ids, test_ids = get_identifier_array_train_test_split()
    X_train, y_train = drop_by_identifier(X_train, y_train, train_ids,
                                          exclude_by=session_type_mapping[exclude_by])

    model = parse_model_name(model_name=model_params["name"])

    pipeline = make_pipeline(
        MinMaxScaler(),
        model,
        memory=cache_dir)

    cv = RepeatedStratifiedKFold(**cross_val_params)
    grid_search = HpBandSterSearchCV(
        estimator=pipeline, param_distributions=model_params["param_distribution"], cv=cv,
        **grid_search_params)

    grid_result = grid_search.fit(X_train, y_train)

    ex.info[grid_result.scoring] = float(grid_result.best_score_)
    ex.info["best_params"] = grid_result.best_params_

    result_path = Path("search_result.pkl")
    with open(result_path, "wb") as fp:
        pickle.dump(file=fp, obj=grid_result)
    ex.add_artifact(result_path, name="search_result.pkl")
    result_path.unlink()
    rmtree(cache_dir)
