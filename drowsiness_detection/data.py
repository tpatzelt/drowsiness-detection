import json
from pathlib import Path
from random import choice

import dill as pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from drowsiness_detection import config
from drowsiness_detection.helpers import binarize, ArrayWrapper, name_generator

session_type_mapping = dict(a=1, b=2, e=3, s=4)

label_names_dict = {  # num_targets: label_names
    2: ["not drowsy", "drowsy"],
    3: ["active", "neutral", "drowsy"],
    5: ["alert", "awake", "neutral", "slightly drowsy", "drowsy"],
    9: [str(num) for num in range(1, 10)]
}


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
    np.save(sub_target_dir.joinpath("train"), train_identifiers)  # noqa
    np.save(sub_target_dir.joinpath("test"), test_identifiers)  # noqa


def get_train_test_splits(directory: Path = config.PATHS.TRAIN_TEST_SPLIT):
    KSS_THRESHOLD = 7
    test_data, train_data = [], []
    for file in sorted(directory.iterdir()):
        if file.is_dir():
            continue
        arr = np.load(file=file)  # noqa
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
    train_idents = np.load(directory.joinpath("train.npy"))  # noqa
    test_idents = np.load(directory.joinpath("test.npy"))  # noqa
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


def get_feature_data(data_path: Path = config.PATHS.WINDOW_FEATURES):
    """Loads all files under data_path, adds the session type, subject id and kss score as new columns. """
    all_arrays = []
    for feature_file in data_path.iterdir():
        sess_type, subject_id = filename_to_session_type_and_id(feature_file)
        features = np.load(feature_file)  # noqa
        targets = get_kss_labels_for_feature_file(feature_file)
        sess_types = np.repeat(session_type_mapping[sess_type], len(targets))
        subject_ids = np.repeat(subject_id, len(targets))
        merged_arr = np.c_[features, targets, sess_types, subject_ids]
        all_arrays.append(merged_arr)
    return np.concatenate(all_arrays)


def preprocess_feature_data(feature_data: np.ndarray, exclude_sess_type: int, num_targets: int):
    """Preprocessing the feature data includes removing NaNs,
    binarize kss scores and split into features and targets.
    returns subject_data which is an array [:, sess id, subject id]"""
    # col -3 is targets, -2 is sess type and -1 is subject id
    feature_data = np.nan_to_num(feature_data)
    targets = feature_data[:, -3]
    if num_targets == 2:
        KSS_THRESHOLD = 7
        targets = binarize(targets, threshold=KSS_THRESHOLD)
    elif num_targets == 3:
        ALERT = 6
        NEUTRAL = 8
        SLEEPY = 10
        targets = np.digitize(targets, bins=[ALERT, NEUTRAL, SLEEPY])
    elif num_targets == 5:
        first = 3
        second = 5
        third = 7
        forth = 9
        fifth = 10
        targets = np.digitize(targets, bins=[first, second, third, forth, fifth])
    elif num_targets == 9:
        targets = np.digitize(targets, bins=range(1, 10))
    else:
        raise ValueError(f"num targets {num_targets} not supported.")

    feature_data[:, -3] = targets
    # remove one session type
    feature_data = feature_data[feature_data[:, -2] != exclude_sess_type]
    X = feature_data[:, :-3]
    y = feature_data[:, -3]
    subject_data = feature_data[:, -2:]
    return X, y, subject_data


def discretize_labels_by_threshold(targets, num_targets: int):
    if num_targets == 2:
        KSS_THRESHOLD = 7
        targets = binarize(targets, threshold=KSS_THRESHOLD)
    elif num_targets == 3:
        ALERT = 6
        NEUTRAL = 8
        SLEEPY = 10
        targets = np.digitize(targets, bins=[ALERT, NEUTRAL, SLEEPY])
    elif num_targets == 5:
        first = 3
        second = 5
        third = 7
        forth = 9
        fifth = 10
        targets = np.digitize(targets, bins=[first, second, third, forth, fifth])
    elif num_targets == 9:
        targets = np.digitize(targets, bins=range(1, 10))
    else:
        raise ValueError(f"num targets {num_targets} not supported.")
    return targets


def get_data_for_nn(data_path: Path = config.PATHS.WINDOW_DATA):
    """Loads all files under data_path, adds the session type, subject id and kss score as new columns. """
    for feature_file in data_path.iterdir():
        sess_type, subject_id = filename_to_session_type_and_id(feature_file)
        features = np.load(feature_file)  # (355,301,23)  # noqa
        targets = get_kss_labels_for_feature_file(feature_file)  # (355,)
        if features.shape[0] != targets.shape[0]:
            raise ValueError(f"{features.shape} vs {targets.shape}")

        sess_types = np.repeat(session_type_mapping[sess_type], len(targets))
        subject_ids = np.repeat(subject_id, len(targets))
        yield features, targets, sess_types, subject_ids


def preprocess_data_for_nn(data_generator, exclude_sess_type: int, num_targets: int):
    for feature_data, targets, session_types, subject_ids in data_generator:
        targets = discretize_labels_by_threshold(targets=targets, num_targets=num_targets)
        feature_data = np.nan_to_num(feature_data)
        # remove one session type
        session_mask = session_types != exclude_sess_type
        feature_data = feature_data[session_mask, :, :]
        targets = targets[session_mask]
        session_types = session_types[session_mask]
        subject_ids = subject_ids[session_mask]
        # print(f"{feature_data.shape} vs {targets.shape}")
        if feature_data.shape[0] != targets.shape[0]:
            raise ValueError(f"{feature_data.shape} vs {targets.shape}")
        if feature_data.size == 0:
            continue
        yield feature_data, targets, np.c_[session_types, subject_ids]


# %%
def merge_nn_data(data_generator):
    """ Aggregates the data from data_generator into one array."""
    values_per_block = config.PATHS.seconds * config.PATHS.frequency
    Xs, ys = [], []
    subject_data_s = []
    # i = 0
    for X, y, subject_data in data_generator:
        if X.shape[1] < values_per_block:
            num_missing_rows = values_per_block - X.shape[1]
            missing_rows_X = np.stack([X[:, -1, :]] * num_missing_rows, axis=1)
            X = np.concatenate([X, missing_rows_X], axis=1)
        elif X.shape[1] > values_per_block:
            X = X[:, :values_per_block, :]
        Xs.append(X)
        ys.append(y)
        subject_data_s.append(subject_data)
        # if i == 10:
        #     break
        # i += 1
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(subject_data_s)


def load_nn_data(exclude_by: int = 1, data_path: Path = config.PATHS.WINDOW_DATA,
                 num_targets: int = 2):
    data_gen = get_data_for_nn(data_path=data_path)
    data_gen = preprocess_data_for_nn(data_generator=data_gen, exclude_sess_type=exclude_by,
                                      num_targets=num_targets)
    return merge_nn_data(data_generator=data_gen)


def load_experiment_config(experiment_id: int):
    with open(f"../../logs/{experiment_id}/config.json") as fp:
        return json.load(fp)


def load_experiment_best_model(experiment_id: int):
    with open(f"../../logs/{experiment_id}/best_model.pkl", "rb") as fp:
        return pickle.load(fp)


def load_experiment_search_results(experiment_id: int):
    with open(f"../../logs/{experiment_id}/search_result.pkl", "rb") as fp:
        return pickle.load(fp)


def load_experiment_objects(experiment_id: int):
    config = load_experiment_config(experiment_id)
    best_model = load_experiment_best_model(experiment_id)
    search_results = load_experiment_search_results(experiment_id)
    return config, best_model, search_results


def train_test_split_by_subjects(X, y, num_targets, test_size, subject_data):
    if num_targets == 2:
        MIN_LABELS = 0
        MAX_LABELS = 1
    else:
        MIN_LABELS = 1
        MAX_LABELS = num_targets
    train, test = [], []
    train_ids, test_ids = [], []
    train_subject_info, test_subject_info = [], []
    test_labels, train_labels = np.empty(0), np.empty(0)
    if MIN_LABELS == 0:
        bins = np.linspace(MIN_LABELS, MAX_LABELS + 1, MAX_LABELS + 2) - 0.5
    else:
        bins = np.linspace(MIN_LABELS, MAX_LABELS + 1, MAX_LABELS + 1) - 0.5
    subject_ids = np.unique(subject_data[:, -1])
    np.random.shuffle(subject_ids)
    for subject_id in subject_ids:
        mask = subject_data[:, -1] == subject_id
        subject_rows = X[mask]
        labels = y[mask]
        subject_info_rows = subject_data[mask]
        assert len(labels) == len(subject_rows)
        if not train:
            train.append(subject_rows)
            train_labels = np.concatenate([train_labels, labels])
            train_ids.append(subject_id)
            train_subject_info.append(subject_info_rows)
            continue
        if not test:
            test.append(subject_rows)
            test_labels = np.concatenate([test_labels, labels])
            test_ids.append(subject_id)
            test_subject_info.append(subject_info_rows)
            continue
        test_labels_for_dist = np.concatenate([test_labels, labels])
        train_labels_for_dist = np.concatenate([train_labels, labels])
        original_test_hist = np.histogram(test_labels, bins=bins)[0]
        original_train_hist = np.histogram(train_labels, bins=bins)[0]
        test_hist = np.histogram(test_labels_for_dist, bins=bins)[0]
        train_hist = np.histogram(train_labels_for_dist, bins=bins)[0]

        dist_if_train_added = np.linalg.norm(train_hist - original_test_hist)
        dist_if_test_added = np.linalg.norm(test_hist - original_train_hist)
        if dist_if_test_added > dist_if_train_added or (len(test_labels) / len(X)) > test_size:
            train.append(subject_rows)
            train_labels = np.concatenate([train_labels, labels])
            train_ids.append(subject_id)
            train_subject_info.append(subject_info_rows)

        else:
            test.append(subject_rows)
            test_labels = np.concatenate([test_labels, labels])
            test_ids.append(subject_id)
            test_subject_info.append(subject_info_rows)

        assert len(labels) == len(subject_rows)
        assert len(labels) == len(subject_rows)
    train = np.concatenate(train)
    test = np.concatenate(test)
    train_subject_info = np.concatenate(train_subject_info)
    test_subject_info = np.concatenate(test_subject_info)
    # print([x.shape for x in (train, train_labels, test, test_labels)])
    assert len(train) == len(train_labels)
    assert (len(test) == len(test_labels))
    return train, test, train_labels, test_labels, (
        np.array(train_ids).astype(int), np.array(test_ids).astype(int)), (
           train_subject_info, test_subject_info)


def load_preprocessed_train_test_splits(data_path, exclude_sess_type, num_targets, seed, test_size,
                                        split_by_subjects: int = False):
    np.random.seed(seed)
    random.seed(seed)

    data = get_feature_data(data_path=data_path)
    X, y, subject_data = preprocess_feature_data(feature_data=data,
                                                 exclude_sess_type=exclude_sess_type,
                                                 num_targets=num_targets)
    if split_by_subjects:
        X_train, X_test, y_train, y_test, _, (
            train_subject_info, test_subject_info) = train_test_split_by_subjects(X, y,
                                                                                  num_targets=num_targets,
                                                                                  test_size=test_size,
                                                                                  subject_data=subject_data)
        # print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        # print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        return X_train, X_test, y_train, y_test, (train_subject_info, test_subject_info)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=seed)
        return X_train, X_test, y_train, y_test, None


def load_preprocessed_train_val_test_splits(data_path, exclude_sess_type, num_targets, seed,
                                            test_size, split_by_subjects: int = True):
    X_train, X_test, y_train, y_test, (
    train_subject_info, test_subject_info) = load_preprocessed_train_test_splits(
        data_path, exclude_sess_type, num_targets, seed, test_size, split_by_subjects)
    X_train, X_val, y_train, y_val, _, _ = train_test_split_by_subjects(X_train, y_train,
                                                                        num_targets=num_targets,
                                                                        test_size=test_size / (
                                                                                    1 - test_size),
                                                                        subject_data=train_subject_info)
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_preprocessed_train_val_test_splits_nn(data_path, exclude_sess_type, num_targets,
                                               seed, test_size):
    np.random.seed(seed)
    random.seed(seed)
    X, y, subject_data = load_nn_data(data_path=data_path, exclude_by=exclude_sess_type,
                                      num_targets=num_targets)
    X_train, X_test, y_train, y_test, _, (
        train_subject_info, test_subject_info) = train_test_split_by_subjects(X, y,
                                                                              num_targets=num_targets,
                                                                              test_size=test_size,
                                                                              subject_data=subject_data)
    X_train, X_val, y_train, y_val, _, _ = train_test_split_by_subjects(
        X_train, y_train, num_targets=num_targets,
        test_size=test_size / (1 - test_size), subject_data=train_subject_info)

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == '__main__':
    import random

    config.set_paths(30, 10)
    random.seed(42)
    window_files_train_test_split(train_size=4, test_size=1)
