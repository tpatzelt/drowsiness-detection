import numpy as np
from itertools import repeat
import pandas as pd


def add_karolinska_file_to_feature_df(filepath: str, feature_df: pd.DataFrame):
    df = pd.read_csv(filepath)

    # add nearest neighbor response to frames
    begin = 0
    responses = []
    for index, row in df.iterrows():
        frame_end = row["frame_end"]
        response = row["response_karolinska"]
        if index < len(df) - 1:
            next_frame_begin = df.iloc[index + 1]["frame_begin"]
            end = int((frame_end + next_frame_begin) / 2)

        else:
            end = max(feature_df.index) + 1
        some_responses = list(zip(range(begin, end), repeat(response)))
        responses.extend(some_responses)
        begin = end
    response_index, responses = zip(*responses)

    target_df = pd.DataFrame(responses, columns=["karolinska_response_nearest_interpolation"], index=response_index)
    feature_df = feature_df.join(target_df)

    # add linear interpolation
    begin = 0
    responses = []
    for index, row in df.iterrows():
        frame_end = row["frame_end"]
        if index < len(df) - 1:
            next_frame_begin = df.iloc[index + 1]["frame_begin"]
            end = int((frame_end + next_frame_begin) / 2)

        else:
            end = max(feature_df.index) + 1
        if index == df.index[0]:
            first_response = row["response_karolinska"]
            second_response = row["response_karolinska"]
        elif index == df.index[-1]:
            first_response = row["response_karolinska"]
            second_response = row["response_karolinska"]
        else:
            first_response = row["response_karolinska"]
            second_response = df.iloc[index]["response_karolinska"]

        some_responses = np.linspace(first_response, second_response, end - begin, dtype=int).tolist()
        some_responses = list(zip(range(begin, end), some_responses))
        responses.extend(some_responses)
        begin = end
    response_index, responses = zip(*responses)
    target_df = pd.DataFrame(responses, columns=["karolinska_response_linear_interpolation"], index=response_index)
    feature_df = feature_df.join(target_df)

    feature_df["karolinska_response_nearest_interpolation"] = feature_df["karolinska_response_nearest_interpolation"].astype("float").astype("Int8", copy=False)
    feature_df["karolinska_response_linear_interpolation"] = feature_df["karolinska_response_linear_interpolation"].astype("float").astype("Int8", copy=False)

    return feature_df
