from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.model_selection import train_test_split
import pickle
from drowsiness_detection.helpers import spec_to_config_space
from hpbandster_sklearn import HpBandSterSearchCV
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from drowsiness_detection import config
from sklearn.metrics import accuracy_score, roc_auc_score

from drowsiness_detection.data import session_type_mapping, get_feature_data, \
    preprocess_feature_data

ex = Experiment("grid_search_kss")
ex.observers.append(FileStorageObserver(Path(__file__).parent.parent.joinpath("logs")))


@ex.config
def base():
    seed = 123
    test_size = .2
    recording_frequency = None
    window_in_sec = None
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
    model_name = None
    hyperparameter_specs = None

    exclude_by = "a"


@ex.named_config
def logistic_regression():
    grid_search_params = {
        "n_iter": 20,
        "max_budget": 5000,
        "resource_name": "max_iter",
        "resource_type": int
    }
    hyperparameter_specs = [dict(name="UniformFloatHyperparameter",
                                 kwargs=dict(name="C", lower=.001, upper=100, log=True)),
                            dict(name="CategoricalHyperparameter",
                                 kwargs=dict(name="solver", choices=["liblinear"])),
                            dict(name="CategoricalHyperparameter",
                                 kwargs=dict(name="penalty",
                                             choices=["l1", "l2"]))]
    model_name = "LogisticRegression"


@ex.named_config
def random_forest():
    model_name = "RandomForestClassifier"
    grid_search_params = {
        "n_iter": 20,
        "max_budget": 3000,
        "resource_name": "n_estimators",
        "resource_type": int
    }
    hyperparameter_specs = [
        dict(name="CategoricalHyperparameter",
             kwargs=dict(name="criterion", choices=["gini", "entropy"])),
        dict(name="UniformIntegerHyperparameter",
             kwargs=dict(name="max_depth", lower=2, upper=100, log=False)),
        dict(name="CategoricalHyperparameter",
             kwargs=dict(name="max_features", choices=["sqrt", "log2"])),
    ]


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
        grid_search_params: dict, model_name: str, exclude_by: str, hyperparameter_specs: dict,
        test_size, seed):
    # set up global paths and cache dir for pipeline
    config.set_paths(frequency=recording_frequency, seconds=window_in_sec)

    # load model
    model = parse_model_name(model_name=model_name)

    # load data
    data = get_feature_data(data_path=config.PATHS.WINDOW_FEATURES)
    X, y = preprocess_feature_data(feature_data=data,
                                   exclude_sess_type=session_type_mapping[exclude_by])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=seed)

    # normalization
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X=X_train)
    X_test = scaler.transform(X_test)

    # set up hyperparameter tuning
    cv = RepeatedStratifiedKFold(**cross_val_params)
    param_distribution = spec_to_config_space(specs=hyperparameter_specs)
    grid_search = HpBandSterSearchCV(
        estimator=model, param_distributions=param_distribution, cv=cv,
        **grid_search_params)

    # run hyperband
    grid_result = grid_search.fit(X_train, y_train)

    # log best model
    ex.info[grid_result.scoring] = float(grid_result.best_score_)
    ex.info["best_params"] = grid_result.best_params_

    # log metrics on test set
    y_pred = grid_result.best_estimator_.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    y_pred_scores = grid_result.best_estimator_.predict_proba(X_test)[:,
                    1]  # scores need to be probs. for class with greater label, i.e. 1.
    test_roc_auc = roc_auc_score(y_test, y_pred_scores)
    ex.info["test_accuracy"] = float(test_acc)
    ex.info["test_roc_auc"] = test_roc_auc

    # save all search results
    result_path = Path("search_result.pkl")
    with open(result_path, "wb") as fp:
        pickle.dump(file=fp, obj=grid_result)
    ex.add_artifact(result_path, name="search_result.pkl")
    result_path.unlink()
