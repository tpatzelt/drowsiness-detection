from pathlib import Path
from random import choice

import numpy as np
import pandas as pd

from drowsiness_detection import config

with open(config.FEATURE_NAMES_PATH) as fp:
    FEATURE_NAMES = fp.read().split("\n")

def bytes_to_MB(b=int):
    return b / 1024 / 1024


def MB_to_bytes(mb=int):
    return mb * 1024 * 1024


def name_generator(base_name: str) -> str:
    i = 0
    while True:
        yield base_name + "_" + str(i)
        i += 1


def create_emtpy_array_of_max_size(max_size_mb, n_cols):
    max_bytes = MB_to_bytes(mb=max_size_mb)
    max_rows = np.floor(max_bytes / 8 / n_cols)
    return np.empty(shape=(max_rows.astype(int), n_cols))


class ArrayWrapper:
    def __init__(self, directory: Path, filename_generator, max_size_mb=1, n_cols=1):
        self.max_size_mb = max_size_mb
        self.directory = directory
        self.name_gen = filename_generator
        self.array = create_emtpy_array_of_max_size(max_size_mb=max_size_mb, n_cols=n_cols)
        self.i = iter(range(self.array.shape[0]))

    def add(self, row):
        try:
            next_i = next(self.i)
            self.array[next_i] = row
        except StopIteration:
            np.save(file=self.directory.joinpath(next(self.name_gen)), arr=self.array)
            self.array = create_emtpy_array_of_max_size(max_size_mb=self.max_size_mb, n_cols=n_cols)
            self.i = iter(range(self.array.shape[0]))

    def close(self):
        np.save(file=self.directory.joinpath(next(self.name_gen)), arr=self.array[:next(self.i)])


def get_kss_labels_for_feature_file(feature_file_path):
    interpolated_kss_index = 2
    identifier = str(feature_file_path.stem)[-11:]
    for label_file in config.WINDOW_LABEL_PATH.iterdir():
        if identifier in str(label_file):
            return np.load(label_file)[:, interpolated_kss_index]
    else:
        return None


def window_files_train_test_split(
        target_dir: Path = config.DATA_PATH.joinpath("TrainTestSplits"), max_filesize_in_mb: int = 100, n_cols: int = 68, train_size: int = 2,
        test_size: int = 1):
    """ Iterates through all features files under 'config.WINDOW_FEATURES_PATH,
     fetches the label file and writes each row randomly to either the test or
     train set."""
    if not target_dir.exists():
        target_dir.mkdir()
    else:
        raise RuntimeError("directory exists.")
    test_names = name_generator(base_name="test")
    train_names = name_generator(base_name="train")
    test_arrays = [ArrayWrapper(directory=target_dir, filename_generator=test_names, max_size_mb=max_filesize_in_mb, n_cols=n_cols) for _ in range(test_size)]
    train_arrays = [ArrayWrapper(directory=target_dir, filename_generator=train_names, max_size_mb=max_filesize_in_mb, n_cols=n_cols) for _ in
                    range(train_size)]
    all_arrays = (*test_arrays, *train_arrays)

    for feature_file in config.WINDOW_FEATURES_PATH.iterdir():
        features = np.load(feature_file)
        targets = get_kss_labels_for_feature_file(feature_file)
        merged_arr = np.c_[features, targets]
        for row in merged_arr:
            target_array = choice(all_arrays)
            target_array.add(row)
    for arr in all_arrays:
        arr.close()

def get_train_test_splits(directory: Path = config.DATA_PATH.joinpath("TrainTestSplits")):
    test_data, train_data = [], []
    for file in directory.iterdir():
        arr = np.load(file=file)
        if "test" in file.name:
            test_data.append(arr)
        elif "train" in file.name:
            train_data.append(arr)
        else:
            raise RuntimeError(f"Cannot assign {file} to train or test split.")
    return np.concatenate(train_data), np.concatenate(test_data)


def feature_array_to_df(arr: np.ndarray) -> pd.DataFrame:
    with open(config.FEATURE_NAMES_PATH) as fp:
        features_names = fp.read().split("\n")
    return pd.DataFrame(arr, columns=features_names)

if __name__ == '__main__':
    window_files_train_test_split()
