from sklearnex import patch_sklearn

patch_sklearn()
from sacred import SETTINGS
# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv  # noqa
# now you can import normally from model_selection
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.utils import to_categorical
from drowsiness_detection.data import (get_feature_data, preprocess_feature_data,
                                       session_type_mapping)
from sklearn.multioutput import MultiOutputClassifier

from pathlib import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.ensemble import RandomForestClassifier
from drowsiness_detection import config
from drowsiness_detection.models import ThreeDStandardScaler

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
SETTINGS['CAPTURE_MODE'] = 'sys'

log_path = Path('/home/tim/PycharmProjects/drowsiness-detection/logs')
if not log_path.exists():
    raise RuntimeError(f"path {log_path} does not exist.")

seed = 123
test_size = .2
recording_frequency = 30
window_in_sec = 60

exclude_by = "a"
n_jobs = -2
rf_params = {
    "criterion": "entropy",
    "max_depth": 32,
    "max_features": "sqrt",
    "n_estimators": 2187,
    "verbose": 1
}

config.set_paths(frequency=recording_frequency, seconds=window_in_sec)
ex = Experiment("rf_different_kss_dicretization", interactive=False)
ex.observers.append(FileStorageObserver(log_path))


@ex.config
def base():
    num_targets = None


@ex.automain
def run(num_targets):
    # set up global paths and cache dir for pipeline

    # load model
    model = RandomForestClassifier(**rf_params)
    model = MultiOutputClassifier(model, n_jobs=n_jobs)
    scaler = ThreeDStandardScaler()
    # load data
    data = get_feature_data(data_path=config.PATHS.WINDOW_FEATURES)
    X, y = preprocess_feature_data(feature_data=data,
                                   exclude_sess_type=session_type_mapping[exclude_by],
                                   num_targets=num_targets)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=seed)

    X_train = scaler.fit_transform(X=X_train, y=y_train)
    X_test = scaler.transform(X_test)

    y_train = to_categorical(y=y_train - 1, num_classes=num_targets)
    y_test = to_categorical(y=y_test - 1, num_classes=num_targets)
    print(X_train.shape)
    print(y_train.shape)

    model.fit(X=X_train, Y=y_train)

    # log metrics on test set
    train_score = model.score(X_train, y_train)
    ex.info["train_accuracy"] = float(train_score)
    test_score = model.score(X_test, y_test)
    ex.info["test_accuracy"] = float(test_score)

    result_path = Path("best_model.pkl")
    with open(result_path, "wb") as fp:
        pickle.dump(file=fp, obj=model)
    ex.add_artifact(result_path, name="best_model.pkl")
    result_path.unlink()
