from pathlib import Path
from random import choice

import numpy as np
import pandas as pd

from drowsiness_detection import config
from drowsiness_detection.helpers import binarize, ArrayWrapper, name_generator

session_type_mapping = dict(a=1, b=2, e=3, s=4)

with open(config.PATHS.FEATURE_NAMES_FILE) as fp:
    FEATURE_NAMES = fp.read().split("\n")


def filename_to_session_type_and_id(filename: Path):
    a = filename.name
    elements = a.split("_")
    session_type = elements[-2]
    identifier = int(elements[2])
    return session_type, identifier


def get_kss_labels_for_feature_file(feature_file_path):
    interpolated_kss_index = 2
    identifier = str(feature_file_path.stem)[-11:]
    for label_file in config.PATHS.LABEL_DATA.iterdir():
        if identifier in str(label_file):
            return np.load(label_file)[:, interpolated_kss_index]
    else:
        return None


def window_files_train_test_split(
        target_dir: Path = config.PATHS.TRAIN_TEST_SPLIT,
        max_filesize_in_mb: int = 100, n_cols: int = 786, train_size: int = 2,
        test_size: int = 1):
    """ Iterates through all features files under 'config.WINDOW_FEATURES_PATH,
     fetches the label file and writes each row randomly to either the test or
     train set."""
    if not target_dir.exists():
        target_dir.mkdir()
    else:
        raise RuntimeError("directory exists.")
    test_identifiers, train_identifiers = [], []
    test_names = name_generator(base_name="test")
    train_names = name_generator(base_name="train")
    test_arrays = [ArrayWrapper(directory=target_dir, filename_generator=test_names,
                                max_size_mb=max_filesize_in_mb, n_cols=n_cols) for _ in
                   range(test_size)]
    train_arrays = [ArrayWrapper(directory=target_dir, filename_generator=train_names,
                                 max_size_mb=max_filesize_in_mb, n_cols=n_cols) for _ in
                    range(train_size)]
    all_arrays = (*test_arrays, *train_arrays)

    for feature_file in config.PATHS.WINDOW_FEATURES.iterdir():
        sess_type, subject_id = filename_to_session_type_and_id(feature_file)
        features = np.load(feature_file)
        targets = get_kss_labels_for_feature_file(feature_file)
        merged_arr = np.c_[features, targets]
        for row in merged_arr:
            target_array = choice(all_arrays)
            target_array.add(row)
            if target_array in test_arrays:
                test_identifiers.append((session_type_mapping[sess_type], subject_id))
            else:
                train_identifiers.append((session_type_mapping[sess_type], subject_id))
    for arr in all_arrays:
        arr.close()

    sub_target_dir = target_dir.joinpath("identifiers")
    sub_target_dir.mkdir()
    np.save(sub_target_dir.joinpath("train"), train_identifiers)
    np.save(sub_target_dir.joinpath("test"), test_identifiers)


def get_train_test_splits(directory: Path = config.PATHS.TRAIN_TEST_SPLIT):
    KSS_THRESHOLD = 7
    test_data, train_data = [], []
    for file in sorted(directory.iterdir()):
        if file.is_dir():
            continue
        arr = np.load(file=file)
        if "test" in file.name:
            test_data.append(arr)
        elif "train" in file.name:
            train_data.append(arr)
        else:
            raise RuntimeError(f"Cannot assign {file} to train or test split.")
    train = np.concatenate(train_data)
    test = np.concatenate(test_data)

    # X still contains NaNs
    train = np.nan_to_num(train, nan=-1)
    test = np.nan_to_num(test, nan=-1)

    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]

    # binarize data
    y_train, y_test = binarize(y_train, KSS_THRESHOLD), binarize(y_test, KSS_THRESHOLD)
    return X_train, y_train, X_test, y_test


def get_data_not_splitted(directory: Path = config.PATHS.TRAIN_TEST_SPLIT):
    X_train, y_train, X_test, y_test = get_train_test_splits(directory=directory)
    # split in full data for CV
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    return X, y


def get_identifier_array_train_test_split(
        directory: Path = config.PATHS.SPLIT_IDENTIFIER):
    train_idents = np.load(directory.joinpath("train.npy"))
    test_idents = np.load(directory.joinpath("test.npy"))
    return train_idents, test_idents


def get_identifier_array_not_splitted(
        directory: Path = config.PATHS.SPLIT_IDENTIFIER):
    train_idents, test_idents = get_identifier_array_train_test_split(directory=directory)
    return np.concatenate([train_idents, test_idents])


def feature_array_to_df(arr: np.ndarray) -> pd.DataFrame:
    with open(config.PATHS.FEATURE_NAMES_FILE) as fp:
        feature_names = fp.read().split("\n")
    return pd.DataFrame(arr, columns=feature_names)


def get_session_idx(ids: np.array):
    session_idx = []
    for s_type, s_int in session_type_mapping.items():
        idx = np.squeeze(np.argwhere(ids[:, 0] == s_int))
        session_idx.append((idx, s_type))
    return session_idx


def get_subject_idx(ids: np.array):
    subject_ids = set()
    for file in config.PATHS.WINDOW_FEATURES.iterdir():
        if file.is_dir():
            continue
        s_type, s_id = filename_to_session_type_and_id(file)
        subject_ids.add(s_id)

    session_idx = []
    for s_id in subject_ids:
        idx = np.squeeze(np.argwhere(ids[:, 1] == s_id))
        session_idx.append((idx, s_id))
    return session_idx


def drop_by_identifier(X, y, identifiers: np.ndarray, exclude_by: int):
    if X.shape[0] != identifiers.shape[0] or y.shape[0] != identifiers.shape[0]:
        raise ValueError("Shapes do not match.")
    X = X[identifiers[:, 0] != exclude_by]
    y = y[identifiers[:, 0] != exclude_by]
    return X, y


if __name__ == '__main__':
    import random

    config.set_paths(30, 10)
    random.seed(42)
    window_files_train_test_split(train_size=4, test_size=1)
