import json
from pathlib import Path

import pandas as pd

from drowsiness_detection.blink_features import add_perclos_features_to_df, add_blink_features_by_eye_state
from drowsiness_detection.config import DATA_PATH
from drowsiness_detection.response_features import add_karolinska_file_to_feature_df


def session_file_to_df(filepath: str, filepath_response: str) -> pd.DataFrame:
    print(
        f"Extracting file {filepath} and response file: {filepath_response}.")
    with open(filepath) as fp:
        data = json.loads(fp.read())

    # join data
    df_eye_closure = pd.DataFrame([item["eye_closure"] for item in data])
    df_eye_closure[
        df_eye_closure <
        0] = 0  # some values are negative and need to be set to zero
    df_eye_state = pd.DataFrame([item["eye_state"] for item in data])
    df_closure_and_state = df_eye_closure.join(df_eye_state,
                                               rsuffix="_eye_state",
                                               lsuffix="_eye_closure")

    # add meta data
    filename = Path(filepath).stem
    subject_id, session_id, session_type = filename.split("_")
    df_closure_and_state["subject_id"] = subject_id
    df_closure_and_state["session_id"] = session_id
    df_closure_and_state["session_type"] = session_type

    # assign dtypes
    df_closure_and_state["subject_id"] = df_closure_and_state[
        "subject_id"].astype("float").astype("Int8", copy=False)
    df_closure_and_state["session_id"] = df_closure_and_state[
        "session_id"].astype("float").astype("Int8", copy=False)
    df_closure_and_state["session_type"] = df_closure_and_state[
        "session_type"].apply(lambda x: (ord(x) - 97)).astype("float").astype(
        "Int8", copy=False)
    df_closure_and_state[[
        "combined_eye_state", "left_image_eye_state", "right_image_eye_state"
    ]] = df_closure_and_state[[
        "combined_eye_state", "left_image_eye_state", "right_image_eye_state"
    ]].astype("float").astype("Int8", copy=False)
    df_closure_and_state = add_karolinska_file_to_feature_df(
        filepath=filepath_response, feature_df=df_closure_and_state)
    # create multi-index
    multi_index = pd.MultiIndex.from_product(
        [[filename], df_closure_and_state.index], names=["filename", "frame"])
    df_closure_and_state.index = multi_index
    # df_closure_and_state.dropna(inplace=True)
    return df_closure_and_state


def create_raw_dataset():
    session_identifier = "karolinska"
    files = [str(p) for p in DATA_PATH.joinpath("sleep_alc_labels").iterdir()]

    session_files = sorted([file for file in files if session_identifier in file])

    feature_files = sorted([str(p) for p in DATA_PATH.joinpath("potsdam_aeye_112020").iterdir()])

    # feature_files = ["../data/potsdam_aeye_112020/014_1_b.json"]
    # session_files = ["../data/sleep_alc_labels/014_1_b_karolinska.csv"]
    full_df = pd.concat(
        list(map(session_file_to_df, feature_files, session_files)))
    return full_df


def create_handcrafted_data(interval: int = 60):
    full_df = create_raw_dataset()
    full_df = full_df[full_df["session_type"] != 0]  # use only baseline and sleep-deprived session data
    df_views = []
    for level in full_df.index.unique(level=0):
        df_view = full_df.loc[level].copy()
        df_view = add_perclos_features_to_df(feature_df=df_view)
        df_view = add_blink_features_by_eye_state(feature_df=df_view, interval_in_sec=interval)
        df_view = df_view[['perclos_combined', 'num_blinks',
                           'mean_blink_length', 'mean_opening_velocity', 'mean_closing_velocity',
                           'max_blink_length', 'min_blink_length', 'karolinska_response_linear_interpolation']]
        df_view = df_view.dropna()
        df_views.append(df_view)

    new_df = pd.concat(df_views)
    new_df.reset_index(inplace=True)
    new_df.pop("frame")
    return new_df

def load_handcrafted_data(interval: int = 60):
    filename = DATA_PATH.joinpath(f"sleepiness_data_{interval}s.pkl")
    try:
        df = pd.read_pickle(filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} not available. There are only {[str(name) for name in list(DATA_PATH.iterdir())]}")
    target_df = df.pop('karolinska_response_linear_interpolation')
    return df, target_df