import numpy as np

from drowsiness_detection.data import create_eye_closure_karolinksa_dataset


def calculate_global_mean_and_std():
    lens = []
    means = []
    for full_df in create_eye_closure_karolinksa_dataset():
        closure_signal = full_df["eye_closure"].to_numpy()
        lens.append(np.count_nonzero(~np.isnan(closure_signal)))
        means.append(np.nanmean(closure_signal))
    weighted_means = 0
    for length, mean in zip(lens, means):
        weighted_means += length * mean
    global_mean = np.divide(weighted_means, np.sum(lens))

    lens = []
    means = []
    for full_df in create_eye_closure_karolinksa_dataset():
        closure_signal = full_df["eye_closure"].to_numpy()
        lens.append(np.count_nonzero(~np.isnan(closure_signal)))
        diffs = (closure_signal[~np.isnan(closure_signal)] - global_mean) ** 2
        means.append(np.mean(diffs))
    weighted_means = 0
    for length, mean in zip(lens, means):
        weighted_means += length * mean
    global_var = np.divide(weighted_means, np.sum(lens))
    global_std = np.sqrt(global_var)
    return global_mean, global_std
