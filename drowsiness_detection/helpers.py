from functools import reduce

import coral_ordinal as coral
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import confusion_matrix
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


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,
                 i,
                 format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plot_cm_matrix(model, X_test, y_test, output_kind="float", clip_min=1, clip_max=9, **kwargs):
    if "normalize" in kwargs:
        normalize = kwargs.pop("normalize")
    else:
        normalize = False

    y_pred = model.predict(X_test)

    if output_kind == "float":
        y_pred = np.array(y_pred).astype(int)
    elif output_kind == "digitize":
        y_pred = digitize(y_pred, **kwargs)
    elif output_kind == "one-hot-like":
        y_pred = one_hot_like_to_label(y_pred, **kwargs)
        # print(np.unique(y_pred))
    elif output_kind == "scores":
        y_pred = np.argmax(y_pred, axis=-1)
    elif output_kind == "cum_logits":
        y_pred = ordinal_to_label(y_pred)
    else:
        raise ValueError(f"kind {output_kind} unkown.")
        
    if clip_min is not None and clip_max is not None:
        y_pred = np.clip(y_pred, clip_min, clip_max)

    c_matrix = confusion_matrix(y_test, y_pred)
    plt.figure()
    plot_confusion_matrix(c_matrix,
                          classes=list("123456789"),
                          normalize=normalize)


def plot_model_history(history):
    for key in history.history:
        plt.plot(history.history[key])
    plt.title('model metrics')
    plt.xlabel('epoch')
    plt.legend(list(history.history.keys()), loc='upper left')
    plt.show()


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
