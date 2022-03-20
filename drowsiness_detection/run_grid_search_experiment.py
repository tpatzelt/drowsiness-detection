from sklearnex import patch_sklearn

patch_sklearn()
# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv  # noqa
# now you can import normally from model_selection
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.pipeline import Pipeline
import pickle
from drowsiness_detection.helpers import spec_to_config_space
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from drowsiness_detection import config

from drowsiness_detection.data import session_type_mapping, get_feature_data, \
    preprocess_feature_data

ex = Experiment("grid_search_kss")
ex.observers.append(FileStorageObserver(Path(__file__).parent.parent.joinpath("logs")))


@ex.config
def base():
    seed = 123
    recording_frequency = None
    window_in_sec = None
    n_inner_splits = 3
    n_outer_splits = 3
    grid_search_params = {
        "factor": None,
        "max_resources": None,
        "resource": None,
        "scoring": None,
        "n_jobs": -1,
        "error_score": "raise",
        "verbose": 1
    }
    model_name = None
    hyperparameter_specs = None

    exclude_by = "a"


@ex.named_config
def dummy_classification():
    grid_search_params = {
        "factor": 3,
        "max_resources": 100,  # "auto",
        "resource": "n_samples",
        "scoring": "accuracy",
    }
    hyperparameter_specs = [
        dict(name="CategoricalHyperparameter",
             kwargs=dict(name="classifier__strategy",
                         choices=["most_frequent", "prior",
                                  "stratified", "uniform"])),
    ]
    model_name = "DummyClassifier"


@ex.named_config
def logistic_regression():
    grid_search_params = {
        "factor": 3,
        "max_resources": "auto",
        "resource": "n_samples",
        "scoring": "accuracy",
    }
    hyperparameter_specs = [dict(name="UniformFloatHyperparameter",
                                 kwargs=dict(name="classifier__C", lower=.001, upper=100, log=True)),
                            dict(name="CategoricalHyperparameter",
                                 kwargs=dict(name="classifier__solver", choices=["liblinear"])),
                            dict(name="CategoricalHyperparameter",
                                 kwargs=dict(name="classifier__penalty",
                                             choices=["l1", "l2"]))]
    model_name = "LogisticRegression"


@ex.named_config
def random_forest():
    model_name = "RandomForestClassifier"
    grid_search_params = {
        "factor": 3,
        "max_resources": 3000,
        "resource": 'classifier__n_estimators',
        "scoring": "accuracy",
    }
    hyperparameter_specs = [
        dict(name="CategoricalHyperparameter",
             kwargs=dict(name="classifier__criterion", choices=["gini", "entropy"])),
        dict(name="UniformIntegerHyperparameter",
             kwargs=dict(name="classifier__max_depth", lower=2, upper=100, log=False)),
        dict(name="CategoricalHyperparameter",
             kwargs=dict(name="classifier__max_features", choices=["sqrt", "log2"])),
    ]


def parse_model_name(model_name: str):
    if model_name == "RandomForestClassifier":
        model = RandomForestClassifier()
    elif model_name == "LogisticRegression":
        model = LogisticRegression()
    elif model_name == "DummyClassifier":
        model = DummyClassifier()
    else:
        raise ValueError
    return model


@ex.automain
def run(recording_frequency: int, window_in_sec: int, n_inner_splits: int, n_outer_splits: int,
        grid_search_params: dict, model_name: str, exclude_by: str, hyperparameter_specs: dict,
        seed):
    # set up global paths and cache dir for pipeline
    config.set_paths(frequency=recording_frequency, seconds=window_in_sec)

    # load model
    model = parse_model_name(model_name=model_name)

    # load data
    data = get_feature_data(data_path=config.PATHS.WINDOW_FEATURES)
    X, y = preprocess_feature_data(feature_data=data,
                                   exclude_sess_type=session_type_mapping[exclude_by])

    inner_cv = KFold(n_splits=n_inner_splits, shuffle=True, random_state=seed)
    outer_cv = KFold(n_splits=n_outer_splits, shuffle=True, random_state=seed)

    pipe = Pipeline([("scaler", MinMaxScaler()), ("classifier", model)])
    print(pipe.get_params().keys())
    param_distribution = spec_to_config_space(specs=hyperparameter_specs)
    search = HalvingRandomSearchCV(estimator=pipe,
                                   param_distributions=param_distribution.get_hyperparameters_dict(),
                                   cv=inner_cv, **grid_search_params)
    search.fit(X=X, y=y)


    # log best model
    ex.info["train_" + search.scoring] = float(search.best_score_)
    ex.info["best_params"] = search.best_params_

    # save all search results
    result_path = Path("search_result.pkl")
    with open(result_path, "wb") as fp:
        pickle.dump(file=fp, obj=search)
    ex.add_artifact(result_path, name="search_result.pkl")
    result_path.unlink()

    test_score = cross_val_score(estimator=search, X=X, y=y, scoring=grid_search_params["scoring"],
                                 n_jobs=1, cv=outer_cv)
    ex.info["test_" + search.scoring] = [float(x) for x in test_score]
