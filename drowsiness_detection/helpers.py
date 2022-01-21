from functools import reduce

import coral_ordinal as coral
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


def ordinal_to_label(arr):
    probs = pd.DataFrame(coral.ordinal_softmax(arr).numpy())
    labels_v1 = probs.idxmax(axis=1).values
    return labels_v1
