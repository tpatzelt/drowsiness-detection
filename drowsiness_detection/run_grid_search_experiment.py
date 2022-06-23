from sklearnex import patch_sklearn

patch_sklearn()
from sacred import SETTINGS
# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv  # noqa
# now you can import normally from model_selection
from sklearn.model_selection import HalvingRandomSearchCV, RandomizedSearchCV, train_test_split, \
    GridSearchCV, PredefinedSplit
import numpy as np
from typing import Tuple
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier
import dill as pickle
from drowsiness_detection.data import (session_type_mapping,
                                       load_preprocessed_train_val_test_splits,
                                       load_preprocessed_train_val_test_splits_nn)
from drowsiness_detection.helpers import spec_to_config_space
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from drowsiness_detection import config
from drowsiness_detection.models import build_dummy_tf_classifier, ThreeDStandardScaler, \
    build_dense_model, build_lstm_model, build_cnn_model
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
SETTINGS['CAPTURE_MODE'] = 'sys'

ex = Experiment("grid_search_kss")
ex.observers.append(FileStorageObserver(Path(__file__).parent.parent.joinpath("logs")))


@ex.config
def base():
    seed = 45
    test_size = .2
    model_selection_name = "halving-random"
    scaler_name = ""
    scaler_params = {}

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
    nn_experiment = False
    feature_col_indices = None


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
    feature_col_indices = (5, 8, 9, 14, 15, 16, 19)
    nn_experiment = True
    model_selection_name = "random"
    scaler_name = "3D-standard"
    grid_search_params = {
        "scoring": "accuracy",
        "n_iter": 2,
        "return_train_score": True,
    }
    model_init_params = {"input_shape": (20, 300, len(feature_col_indices))}
    fit_params = {"classifier__epochs": 1}
    hyperparameter_specs = [
        dict(name="CategoricalHyperparameter",
             kwargs=dict(name="classifier__activation",
                         choices=["softmax", "relu"])),
        dict(name="CategoricalHyperparameter",
             kwargs=dict(name="classifier__optimizer",
                         choices=["adam"])),
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
        "n_iter": 300

    }
    scaler_name = "standard"
    hyperparameter_specs = [
        dict(name="CategoricalHyperparameter",
             # kwargs=dict(name="classifier__criterion", choices=["gini", "entropy"])),
             kwargs=dict(name="classifier__criterion", choices=["entropy"])),
        dict(name="UniformIntegerHyperparameter",
             kwargs=dict(name="classifier__max_depth", lower=2, upper=60, log=False)),
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
             kwargs=dict(name="classifier__min_samples_split", lower=0.005, upper=0.06, log=False)),
    ]
    model_init_params = {}


@ex.named_config
def dense_nn():
    nn_experiment = True
    model_selection_name = "random"
    model_name = "DenseNN"
    grid_search_params = {
        "scoring": "accuracy",
        "return_train_score": True,
        "n_iter": 5,
        "n_jobs": 1,
    }
    fit_params = {"classifier__epochs": 2, "classifier__batch_size": 10}
    model_init_params = {"input_shape": (20, 300, 7)}
    scaler_name = "3D-standard"
    scaler_params = {"feature_axis": -1}

    hyperparameter_specs = [
        dict(name="UniformIntegerHyperparameter",
             kwargs=dict(name="classifier__num_hidden", lower=32, upper=2024, log=False)),
        # kwargs = dict(name="classifier__num_hidden", lower=2, upper=3, log=False)),
    ]
    feature_col_indices = (5, 8, 9, 14, 15, 16, 19)


@ex.named_config
def cnn():
    nn_experiment = True
    model_selection_name = "random"
    model_name = "CNN"
    grid_search_params = {
        "scoring": "accuracy",
        "return_train_score": True,
        "n_iter": 100,
        "n_jobs": 1,
    }
    fit_params = {"classifier__epochs": 25, "classifier__batch_size": 20, 'classifier__verbose': 0}
    model_init_params = {"input_shape": (20, 1800, 7)}
    scaler_name = "3D-standard"
    scaler_params = {"feature_axis": -1}

    hyperparameter_specs = [
        dict(name="UniformIntegerHyperparameter",
             kwargs=dict(name="classifier__num_filters", lower=16, upper=128, log=False)),
        dict(name="CategoricalHyperparameter",
             kwargs=dict(name="classifier__kernel_size", choices=[3, 5])),
        dict(name="CategoricalHyperparameter",
             kwargs=dict(name="classifier__stride", choices=[3, 5])),
        dict(name="CategoricalHyperparameter",
             kwargs=dict(name="classifier__pooling", choices=["average", "max"])),
        dict(name="UniformIntegerHyperparameter",
             kwargs=dict(name="classifier__num_conv_layers", lower=1, upper=3, log=False)),
        dict(name="UniformFloatHyperparameter",
             kwargs=dict(name="classifier__dropout_rate", lower=0.2, upper=.6, log=False)),
    ]
    feature_col_indices = (5, 8, 9, 14, 15, 16, 19)


@ex.named_config
def lstm():
    nn_experiment = True
    model_selection_name = "random"
    model_name = "LSTM"
    grid_search_params = {
        "scoring": "accuracy",
        "return_train_score": True,
        "n_iter": 1,
        "n_jobs": 1,
    }
    fit_params = {"classifier__epochs": 5, "classifier__batch_size": 30, 'classifier__verbose': 1}
    model_init_params = {"input_shape": (20, 1800, 7)}
    scaler_name = "3D-standard"
    scaler_params = {"feature_axis": -1}
    hyperparameter_specs = [
        dict(name="UniformIntegerHyperparameter",
             kwargs=dict(name="classifier__lstm_units", lower=8, upper=128, log=False)),
        # dict(name="UniformIntegerHyperparameter",
        #      kwargs=dict(name="classifier__lstm2_units", lower=8, upper=128, log=False)),
        dict(name="UniformFloatHyperparameter",
             kwargs=dict(name="classifier__dropout_rate", lower=0.3, upper=.5, log=False)),
        dict(name="UniformIntegerHyperparameter",  # inclusive interval
             kwargs=dict(name="classifier__num_lstm_layers", lower=1, upper=3, log=False)),
        dict(name="UniformFloatHyperparameter",
             kwargs=dict(name="classifier__learning_rate", lower=0, upper=0.05, log=False)),
    ]
    feature_col_indices = (5, 8, 9, 14, 15, 16, 19)


@ex.named_config
def minirocket():
    nn_experiment = True
    model_selection_name = "random"
    model_name = "MINIROCKET"
    grid_search_params = {
        "scoring": "accuracy",
        "return_train_score": True,
        "n_iter": 1,
        "n_jobs": 1,
    }
    fit_params = {}
    model_init_params = {"input_shape": (20, 1800, 7)}
    scaler_name = "3D-standard"
    scaler_params = {"feature_axis": -1}
    hyperparameter_specs = [
        dict(name="UniformFloatHyperparameter",
             kwargs=dict(name="classifier__alpha", lower=0, upper=100, log=True)),
    ]
    feature_col_indices = (5, 8, 9, 14, 15, 16, 19)


def parse_model_name(model_name: str, model_init_params={}):
    if model_name == "RandomForestClassifier":
        model = RandomForestClassifier()
    elif model_name == "LogisticRegression":
        model = LogisticRegression()
    elif model_name == "DummyClassifier":
        model = DummyClassifier()
    elif model_name == "DummyTFClassifier":
        def model_fn(activation, optimizer):
            return build_dummy_tf_classifier(activation=activation, optimizer=optimizer,
                                             **model_init_params)

        model = KerasClassifier(build_fn=model_fn)
    elif model_name == "DenseNN":
        def model_fn(num_hidden):
            return build_dense_model(num_hidden=num_hidden, **model_init_params)

        model = KerasClassifier(build_fn=model_fn)
    elif model_name == "CNN":
        def model_fn(kernel_size, stride, num_filters, num_conv_layers, pooling, dropout_rate):
            return build_cnn_model(kernel_size=kernel_size, stride=stride,
                                   num_filters=num_filters, num_conv_layers=num_conv_layers,
                                   pooling=pooling, dropout_rate=dropout_rate,
                                   **model_init_params)

        model = KerasClassifier(build_fn=model_fn)
    elif model_name == "LSTM":
        def model_fn(lstm_units, dropout_rate, num_lstm_layers, learning_rate):
            return build_lstm_model(lstm_units=lstm_units,
                                    dropout_rate=dropout_rate, num_lstm_layers=num_lstm_layers,
                                    learning_rate=learning_rate,
                                    **model_init_params)

        model = KerasClassifier(build_fn=model_fn)
    elif model_name == "MINIROCKET":
        model = RidgeClassifier()
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


def parse_scaler_name(scaler_name: str, scaler_params: dict):
    if scaler_name == "min-max":
        scaler = MinMaxScaler(**scaler_params)
    elif scaler_name == "standard":
        scaler = StandardScaler(**scaler_params)
    elif scaler_name == "3D-standard":
        scaler = ThreeDStandardScaler(**scaler_params)
    else:
        raise ValueError
    return scaler


def load_experiment_data(feature_col_indices, seed,
                         use_dummy_data, test_size, nn_experiment, exclude_by, num_targets,
                         split_by_subjects):
    if use_dummy_data:
        num_samples = 200
        num_feature_cols = len(feature_col_indices)
        X = np.random.random(num_samples * 300 * num_feature_cols).reshape(
            (num_samples, 300, num_feature_cols))
        y = np.concatenate((np.zeros((num_samples // 2)), np.ones((num_samples // 2))))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                          test_size=test_size * (1 - test_size),
                                                          random_state=seed)
    else:
        if nn_experiment:
            X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_train_val_test_splits_nn(
                data_path=config.PATHS.WINDOW_DATA,
                exclude_sess_type=session_type_mapping[exclude_by],
                num_targets=num_targets, seed=seed, test_size=test_size,
                feature_col_indices=feature_col_indices)

        else:
            X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_train_val_test_splits(
                data_path=config.PATHS.WINDOW_FEATURES,
                exclude_sess_type=session_type_mapping[exclude_by],
                num_targets=num_targets, seed=seed, test_size=test_size,
                split_by_subjects=split_by_subjects)

    # need to have extra validation set so that we have the indices of the subjects,
    # then put together with training set
    split_idx = np.concatenate([np.ones(len(X_val)), np.repeat(-1, len(X_train))])
    X_train = np.concatenate([X_val, X_train])
    y_train = np.concatenate([y_val, y_train])
    del X_val, y_val
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return X_train, X_test, y_train, y_test, split_idx


def init_model_selection(model_selection_name, estimator, param_distribution, cv,
                         grid_search_params, seed):
    model_selection = parse_model_selection_name(model_selection_name=model_selection_name)
    if model_selection is GridSearchCV:
        ## not tested
        search = model_selection(estimator=estimator,
                                 param_grid=param_distribution.sample_configuration(
                                     grid_search_params.pop("n_iter")),
                                 cv=cv, **grid_search_params)
    else:
        search = model_selection(estimator=estimator,
                                 param_distributions=param_distribution.get_hyperparameters_dict(),
                                 cv=cv, **grid_search_params, random_state=seed)
    return search


def save_search_results(search):
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


def add_callbacks_to_fit_params(fit_params, validation_data):
    filename = f"{config.SOURCES_ROOT_PATH.parent}/logs/{ex.current_run._id}/train_history.csv"
    history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=False)
    fit_params = fit_params.copy()
    fit_params["classifier__callbacks"] = [history_logger]

    fit_params["classifier__validation_data"] = validation_data
    return fit_params


def log_train_and_test_score(estimator, scoring, X_train, X_test, y_train, y_test):
    train_score = estimator.score(X_train, y_train)
    ex.info["train_" + scoring] = float(train_score)
    test_score = estimator.score(X_test, y_test)
    ex.info["test_" + scoring] = float(test_score)


def save_best_model(estimator):
    if isinstance(estimator.named_steps["classifier"], KerasClassifier):
        result_path = Path(ex.observers[0].dir).joinpath("best_model")
        result_path.mkdir()
        estimator.named_steps["classifier"].model.save(result_path)
    else:
        result_path = Path("best_model.pkl")
        with open(result_path, "wb") as fp:
            pickle.dump(file=fp, obj=estimator)
        ex.add_artifact(result_path, name="best_model.pkl")
        result_path.unlink()


@ex.automain
def run(recording_frequency: int, window_in_sec: int, model_selection_name: str, scaler_name: str,
        grid_search_params: dict, model_name: str, exclude_by: str, hyperparameter_specs: dict,
        seed, test_size: float, n_splits: int, num_targets: int, use_dummy_data: bool,
        split_by_subjects: bool, fit_params: dict, model_init_params: dict, nn_experiment: bool,
        feature_col_indices: Tuple, scaler_params: dict):
    config.set_paths(frequency=recording_frequency, seconds=window_in_sec)
    print(f"Starting experiment on {window_in_sec} sec data with {num_targets} targets.")

    # load model
    model = parse_model_name(model_name=model_name, model_init_params=model_init_params)
    scaler = parse_scaler_name(scaler_name=scaler_name, scaler_params=scaler_params)
    pipeline_steps = [("scaler", scaler), ("classifier", model)]
    if model_name == "MINIROCKET":
        pipeline_steps.insert(1, ("minirocket", MiniRocketMultivariate()))
    pipe = Pipeline(pipeline_steps)
    param_distribution = spec_to_config_space(specs=hyperparameter_specs)

    # load data
    print("loading data")
    X_train, X_test, y_train, y_test, split_idx = load_experiment_data(
        feature_col_indices=feature_col_indices, seed=seed, use_dummy_data=use_dummy_data,
        test_size=test_size, nn_experiment=nn_experiment, exclude_by=exclude_by,
        num_targets=num_targets, split_by_subjects=split_by_subjects)

    cv = PredefinedSplit(test_fold=split_idx)
    search = init_model_selection(model_selection_name, estimator=pipe,
                                  param_distribution=param_distribution, cv=cv,
                                  grid_search_params=grid_search_params, seed=seed)
    print("starting hyperparameter search")
    search.fit(X=X_train, y=y_train, **fit_params)
    print("finished finding the best hyperparameters")
    # log scores of best model
    save_search_results(search=search)
    best_params = search.best_params_.copy()
    del search

    # initialize estimator with best params and retrain on complete dataset
    if nn_experiment:
        fit_params = add_callbacks_to_fit_params(fit_params=fit_params,
                                                 validation_data=(X_test, y_test))
    # train model on entire dataset
    print("start training best model on complete training and validation data")
    new_pipe: Pipeline = pipe.set_params(**best_params)  # noqa
    new_pipe.fit(X=X_train, y=y_train, **fit_params)

    # log metrics on test and train set
    print("log scores and training and test set")
    log_train_and_test_score(estimator=new_pipe, scoring=grid_search_params["scoring"],
                             X_train=X_train, X_test=X_test,
                             y_train=y_train, y_test=y_test)

    # save best model instance
    save_best_model(estimator=new_pipe)
