from sklearnex import patch_sklearn

patch_sklearn()
from sacred import SETTINGS
# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv  # noqa
# now you can import normally from model_selection
from sklearn.model_selection import HalvingRandomSearchCV, RandomizedSearchCV, train_test_split, \
    GridSearchCV, PredefinedSplit

import numpy as np
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier
import dill as pickle
from drowsiness_detection.data import (session_type_mapping,
                                       load_preprocessed_train_val_test_splits)
from drowsiness_detection.helpers import spec_to_config_space
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from drowsiness_detection import config
from drowsiness_detection.models import build_dummy_tf_classifier, ThreeDStandardScaler, \
    build_dense_model

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
SETTINGS['CAPTURE_MODE'] = 'sys'

ex = Experiment("grid_search_kss")
ex.observers.append(FileStorageObserver(Path(__file__).parent.parent.joinpath("logs")))


@ex.config
def base():
    seed = 123
    test_size = .3
    model_selection_name = "halving-random"
    scaler_name = ""
    recording_frequency = None
    window_in_sec = None
    n_splits = 10
    grid_search_params = {
        "scoring": None,
        "n_jobs": -1,
        "error_score": 0,
        "verbose": 1,
        "refit": False,
        # "return_train_score": True

    }
    fit_params = {}
    model_name = None
    hyperparameter_specs = None

    exclude_by = "a"
    num_targets = 2
    use_dummy_data = False
    split_by_subjects = True


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
                         choices=["most_frequent",
                                  "prior",
                                  "stratified",
                                  "uniform"
                                  ])),
    ]
    model_name = "DummyClassifier"
    scaler_name = "3D-standard"


@ex.named_config
def dummy_tf_classification():
    model_selection_name = "random"
    scaler_name = "3D-standard"
    grid_search_params = {
        "scoring": "accuracy",
        "n_iter": 2
    }
    hyperparameter_specs = [
        dict(name="CategoricalHyperparameter",
             kwargs=dict(name="classifier__activation",
                         choices=["softmax", "relu"])),
        dict(name="CategoricalHyperparameter",
             kwargs=dict(name="classifier__batch_size",
                         choices=[10, 20])),
        dict(name="CategoricalHyperparameter",
             kwargs=dict(name="classifier__optimizer",
                         choices=["adam"])),
        dict(name="CategoricalHyperparameter",
             kwargs=dict(name="classifier__input_shape",
                         choices=["1,300,23"]))
    ]
    model_name = "DummyTFClassifier"


@ex.named_config
def logistic_regression():
    grid_search_params = {
        "factor": 3,
        "max_resources": "auto",
        "resource": "n_samples",
        "scoring": "accuracy",
    }
    hyperparameter_specs = [dict(name="UniformFloatHyperparameter",
                                 kwargs=dict(name="classifier__C", lower=.001, upper=100,
                                             log=True)),
                            dict(name="CategoricalHyperparameter",
                                 kwargs=dict(name="classifier__solver", choices=["liblinear"])),
                            dict(name="CategoricalHyperparameter",
                                 kwargs=dict(name="classifier__penalty",
                                             choices=["l1", "l2"]))]
    model_name = "LogisticRegression"


@ex.named_config
def random_forest():
    model_selection_name = "random"
    model_name = "RandomForestClassifier"
    grid_search_params = {
        # "factor": 3,
        # "max_resources": 1000,
        # "resource": 'classifier__n_estimators',
        "scoring": "accuracy",
        "return_train_score": True,
        "n_iter": 100

    }
    scaler_name = "standard"
    hyperparameter_specs = [
        dict(name="CategoricalHyperparameter",
             # kwargs=dict(name="classifier__criterion", choices=["gini", "entropy"])),
             kwargs=dict(name="classifier__criterion", choices=["entropy"])),
        dict(name="UniformIntegerHyperparameter",
             kwargs=dict(name="classifier__max_depth", lower=2, upper=150, log=False)),
        # dict(name="CategoricalHyperparameter",
        #      kwargs=dict(name="classifier__max_depth", choices=[max_depth])),
        dict(name="CategoricalHyperparameter",
             # kwargs=dict(name="classifier__max_features", choices=["sqrt", "log2"])),
             kwargs=dict(name="classifier__max_features", choices=["sqrt"])),
        dict(name="CategoricalHyperparameter",
             kwargs=dict(name="classifier__n_estimators", choices=[512])),
        dict(name="CategoricalHyperparameter",
             kwargs=dict(name="classifier__class_weight", choices=["balanced"])),
        # dict(name="CategoricalHyperparameter",
        #      kwargs=dict(name="classifier__max_samples", choices=[.6,.8])),
        # dict(name="CategoricalHyperparameter",
        #      kwargs=dict(name="classifier__min_samples_split", choices=[0.015, 0.03]))
        dict(name="UniformFloatHyperparameter",
             kwargs=dict(name="classifier__min_samples_split", lower=0.015, upper=0.04, log=False)),
    ]


@ex.named_config
def dense_nn():
    use_dummy_data = True
    model_selection_name = "random"
    model_name = "DenseNN"
    grid_search_params = {
        "scoring": "accuracy",
        "return_train_score": True,
        "n_iter": 2
    }
    fit_params = {"classifier__epochs": 1, "classifier__batch_size": 128}
    model_init_params = {"input_shape": (20, 300, 23)}
    scaler_name = "3D-standard"
    hyperparameter_specs = [
        dict(name="UniformIntegerHyperparameter",
             kwargs=dict(name="classifier__num_hidden", lower=32, upper=2024, log=False)),
    ]


def parse_model_name(model_name: str, model_init_params={}):
    if model_name == "RandomForestClassifier":
        model = RandomForestClassifier()
    elif model_name == "LogisticRegression":
        model = LogisticRegression()
    elif model_name == "DummyClassifier":
        model = DummyClassifier()
    elif model_name == "DummyTFClassifier":
        model = KerasClassifier(build_fn=build_dummy_tf_classifier)
    elif model_name == "DenseNN":
        def model_fn(num_hidden):
            return build_dense_model(num_hidden=num_hidden, **model_init_params)

        model = KerasClassifier(build_fn=model_fn)
    else:
        raise ValueError
    return model


def parse_model_selection_name(model_selection_name: str):
    if model_selection_name == "random":
        model_selection = RandomizedSearchCV
    elif model_selection_name == "halving-random":
        model_selection = HalvingRandomSearchCV
    elif model_selection_name == "grid":
        model_selection = GridSearchCV
    else:
        raise ValueError
    return model_selection


def parse_scaler_name(scaler_name: str):
    if scaler_name == "min-max":
        scaler = MinMaxScaler()
    elif scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "3D-standard":
        scaler = ThreeDStandardScaler()
    else:
        raise ValueError
    return scaler


@ex.automain
def run(recording_frequency: int, window_in_sec: int, model_selection_name: str, scaler_name: str,
        grid_search_params: dict, model_name: str, exclude_by: str, hyperparameter_specs: dict,
        seed, test_size: float, n_splits: int, num_targets: int, use_dummy_data: bool,
        split_by_subjects: bool, fit_params: dict, model_init_params: dict):
    # set up global paths and cache dir for pipeline
    config.set_paths(frequency=recording_frequency, seconds=window_in_sec)

    print(f"Starting experiment on {window_in_sec} sec data with {num_targets} targets.")

    # load model
    model = parse_model_name(model_name=model_name, model_init_params=model_init_params)
    scaler = parse_scaler_name(scaler_name=scaler_name)
    # load data
    if use_dummy_data:
        num_samples = 200
        X = np.random.random(num_samples * 300 * 23).reshape((num_samples, 300, 23))
        y = np.concatenate((np.zeros((num_samples // 2)), np.ones((num_samples // 2))))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                          test_size=test_size * (1 - test_size),
                                                          random_state=seed)
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_train_val_test_splits(
            data_path=config.PATHS.WINDOW_FEATURES,
            exclude_sess_type=session_type_mapping[exclude_by],
            num_targets=num_targets, seed=seed, test_size=test_size,
            split_by_subjects=split_by_subjects)

    # need to have extra validation set so that we have the indices of the subjects,
    # then put together with training set
    split_idx = np.concatenate([np.ones(len(X_val)), np.repeat(-1, len(X_test))])
    X_train = np.concatenate([X_val, X_train])
    y_train = np.concatenate([y_val, y_train])
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    cv = PredefinedSplit(test_fold=split_idx)

    pipe = Pipeline([("scaler", scaler), ("classifier", model)])
    param_distribution = spec_to_config_space(specs=hyperparameter_specs)

    model_selection = parse_model_selection_name(model_selection_name=model_selection_name)
    if model_selection is GridSearchCV:
        raise NotImplementedError("Grid search does not work with hyperparameter dict.")
    else:
        search = model_selection(estimator=pipe,
                                 param_distributions=param_distribution.get_hyperparameters_dict(),
                                 cv=cv, **grid_search_params, random_state=seed)
    search.fit(X=X_train, y=y_train, **fit_params)

    # log scores of best model
    ex.info["best_cv_test_" + search.scoring] = float(
        search.cv_results_["mean_test_score"][search.best_index_])
    ex.info["best_cv_train_" + search.scoring] = float(
        search.cv_results_["mean_train_score"][search.best_index_])
    ex.info["best_params"] = search.best_params_

    # save all search results
    result_path = Path("search_result.pkl")
    with open(result_path, "wb") as fp:
        pickle.dump(file=fp, obj=search)
    ex.add_artifact(result_path, name="search_result.pkl")
    result_path.unlink()

    # train model on entire dataset
    new_pipe: Pipeline = pipe.set_params(**search.best_params_)  # noqa
    new_pipe.fit(X=X_train, y=y_train)

    # log metrics on test and train set
    train_score = new_pipe.score(X_train, y_train)
    ex.info["train_" + search.scoring] = float(train_score)
    test_score = new_pipe.score(X_test, y_test)
    ex.info["test_" + search.scoring] = float(test_score)

    # save all search results
    if isinstance(new_pipe.named_steps["classifier"], KerasClassifier):
        result_path = Path(ex.observers[0].dir).joinpath("best_model")
        result_path.mkdir()
        new_pipe.named_steps["classifier"].model.save(result_path)
    else:
        result_path = Path("best_model.pkl")
        with open(result_path, "wb") as fp:
            pickle.dump(file=fp, obj=new_pipe)
        ex.add_artifact(result_path, name="best_model.pkl")
        result_path.unlink()
