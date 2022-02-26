from functools import reduce
from pathlib import Path

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K


def print_nan_intersections(df: pd.DataFrame):
    nan_indices = dict()
    for column in df.columns:
        nan_indices[column] = set(df.loc[df[column].isna()].index)

    for column in df.columns:
        nan_indices_copy = nan_indices.copy()
        col_indices = set(nan_indices_copy.pop(column))
        other_indices = reduce(lambda x, y: set(x) | set(y), nan_indices_copy.values())
        unique_nans = col_indices - other_indices
        print(f"Column {column} has {len(unique_nans)} nans that appear in no other column.")


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def digitize(arr, shift: float = .5, label_start_index: int = 1):
    bins = [x + shift for x in range(1, 9)]
    return np.digitize(arr, bins=bins) + label_start_index


def one_hot_like_to_label(arr, threshold=.5):
    threshold = keras.backend.constant(threshold)
    y_pred_sum = keras.backend.sum(keras.backend.cast(keras.backend.greater(
        arr, threshold), dtype=tf.int64), axis=1)
    return y_pred_sum


def label_to_one_hot_like(arr, k=9):
    arr = np.squeeze(arr)
    one_hots = np.zeros(shape=(len(arr), k - 1), dtype=int)
    for i, element in enumerate(arr):
        one_hots[i, :element] = 1
    return one_hots



def binarize(arr: np.ndarray, threshold: float) -> np.ndarray:
    return (arr < threshold).astype(int)


class ArrayWrapper:
    def __init__(self, directory: Path, filename_generator, max_size_mb=1, n_cols=1):
        self.max_size_mb = max_size_mb
        self.directory = directory
        self.name_gen = filename_generator
        self.array = create_emtpy_array_of_max_size(max_size_mb=max_size_mb, n_cols=n_cols)
        self.i = iter(range(self.array.shape[0]))
        self.n_cols = n_cols

    def add(self, row):
        try:
            next_i = next(self.i)
            self.array[next_i] = row
        except StopIteration:
            np.save(file=self.directory.joinpath(next(self.name_gen)), arr=self.array)
            self.array = create_emtpy_array_of_max_size(max_size_mb=self.max_size_mb,
                                                        n_cols=self.n_cols)
            self.i = iter(range(self.array.shape[0]))

    def close(self):
        np.save(file=self.directory.joinpath(next(self.name_gen)), arr=self.array[:next(self.i)])


def create_emtpy_array_of_max_size(max_size_mb, n_cols):
    max_bytes = MB_to_bytes(mb=max_size_mb)
    max_rows = np.floor(max_bytes / 8 / n_cols)
    return np.empty(shape=(max_rows.astype(int), n_cols))


def bytes_to_MB(b=int):
    return b / 1024 / 1024


def MB_to_bytes(mb=int):
    return mb * 1024 * 1024


def name_generator(base_name: str) -> str:
    i = 0
    while True:
        yield base_name + "_" + str(i)
        i += 1


def spec_to_config_space(specs):
    cs = CS.ConfigurationSpace()
    for spec in specs:
        to_add = getattr(CSH, spec["name"])
        cs.add_hyperparameter(to_add(**spec["kwargs"]))
    return cs
