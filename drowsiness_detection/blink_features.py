import json
from pathlib import Path
import pandas as pd
from functools import reduce
from itertools import repeat
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter



def add_blink_features_by_eye_state(feature_df: pd.DataFrame, interval_in_sec: int = 60, fps: int = 30):
    
    def generate_blink_property_dict(both_df: pd.DataFrame, interval_in_sec: int = 60, fps: int = 30):
        blink_idx = set()
        blinks = dict()

        min_closure_for_blink = .7

        for closed_idx in both_df[both_df["combined_eye_state"] == 1].index:
            if closed_idx in blink_idx:
                continue
            try:
                if pd.isna(both_df.loc[closed_idx]["combined_eye_closure"]):
                    continue
            except ValueError:
                print(both_df.loc[closed_idx]["combined_eye_closure"])
                raise TypeError(str(closed_idx))
            # find index of max closed frame
            if closed_idx > 0:
                while pd.notna(both_df.loc[closed_idx - 1][ "combined_eye_closure"]) and \
                        (both_df.loc[closed_idx - 1][ "combined_eye_closure"] >= both_df.loc[closed_idx]["combined_eye_closure"]):
                    closed_idx -= 1
                while pd.notna(both_df.loc[closed_idx + 1][ "combined_eye_closure"]) and \
                    (both_df.loc[closed_idx + 1][ "combined_eye_closure"] >= both_df.loc[closed_idx]["combined_eye_closure"]):
                    closed_idx += 1

            if both_df.loc[closed_idx]["combined_eye_closure"] < min_closure_for_blink:
                continue

            #find start of blink
            start_idx = closed_idx

            # decrease start idx until its under blink threshold, then continue until value increases again
            if start_idx >= 1:
                while pd.notna(both_df.loc[start_idx - 1]["combined_eye_closure"]) and \
                        (both_df.loc[start_idx - 1]["combined_eye_closure"] <= both_df.loc[start_idx]["combined_eye_closure"] or \
                        both_df.loc[start_idx]["combined_eye_closure"] > min_closure_for_blink):
                        start_idx -= 1

            # find end of blink
            end_idx = closed_idx
            # increase end idx until value under blink threshold, then continue until value decreases no more
            while pd.notna(both_df.loc[end_idx + 1]["combined_eye_closure"]) and (both_df.loc[end_idx + 1]["combined_eye_closure"] <= both_df.loc[end_idx]["combined_eye_closure"] or \
                both_df.loc[end_idx]["combined_eye_closure"] > min_closure_for_blink):
                end_idx += 1

            closed_idx = both_df["combined_eye_closure"][start_idx:end_idx + 1].idxmax()

            blink_idx = blink_idx.union({idx for idx in range(start_idx, end_idx + 1)})

            opening_distance = both_df.loc[closed_idx]["combined_eye_closure"] - both_df.loc[end_idx]["combined_eye_closure"]
            opening_time = end_idx - closed_idx

            if opening_time == 0:
                opening_velocity = np.nan
            else:
                opening_velocity = opening_distance / opening_time

            closing_distance = both_df.loc[closed_idx]["combined_eye_closure"] - both_df.loc[start_idx]["combined_eye_closure"]
            closing_time = closed_idx - start_idx
            if closing_time == 0:
                closing_velocity = np.nan
            else:
                closing_velocity = closing_distance / closing_time


            # if closed_idx == 12741:
            #     print(both_df.loc[start_idx, "combined_eye_closure"], both_df.loc[closed_idx, "combined_eye_closure"],both_df.loc[end_idx, "combined_eye_closure"])
            #     print(opening_distance, opening_time)
            #     print(closing_distance, closing_time)
            properties = {"start": start_idx,
                          "max_closed": closed_idx,
                          "end": end_idx,
                          "opening_velocity": opening_velocity,
                          "closing_velocity": closing_velocity,
                          "total_length": end_idx - start_idx}
            blinks[closed_idx] = properties
        return blinks

    # calculate blink properties
    blinks = generate_blink_property_dict(both_df=feature_df, interval_in_sec=interval_in_sec, fps=fps)

    # create array to store average blink length, number of blinks,
    # average opening velocity, average closing velocity
    arr = np.ones(shape=(len(feature_df), 6))
    num_frames = fps * interval_in_sec
    for index in tqdm(range(len(arr))):
        start_idx = index - num_frames
        keys_in_range = [key for key in blinks.keys() if  key > start_idx and key < index]
        num_blinks = len(keys_in_range)
        try:
            max_blink_length = int(np.max(([blinks[key]["total_length"] for key in keys_in_range] + [0])))
            min_blink_length = int(np.min(([blinks[key]["total_length"] for key in keys_in_range] + [0])))
            average_blink_length = float(np.nansum([blinks[key]["total_length"] for key in keys_in_range])) / num_blinks
            average_opening_velocity = float(np.nansum([blinks[key]["opening_velocity"] for key in keys_in_range])) / num_blinks
            average_closing_velocity = float(np.nansum([blinks[key]["closing_velocity"] for key in keys_in_range])) / num_blinks

        except ZeroDivisionError:
            max_blink_length = 0
            min_blink_length = 0
            average_blink_length = 0
            average_opening_velocity = 0
            average_closing_velocity = 0

        arr[index] = np.array([num_blinks, average_blink_length, average_opening_velocity, average_closing_velocity, max_blink_length, min_blink_length])
    feature_df[["num_blinks", "mean_blink_length", "mean_opening_velocity", "mean_closing_velocity", "max_blink_length", "min_blink_length"]] = arr
    
    feature_df["num_blinks"]=feature_df["num_blinks"].astype("Int16", copy=False)

    return feature_df


def add_perclos_features_to_df(feature_df: pd.DataFrame, interval_in_sec: int = 60, fps: int = 30, closed_threshold: float =  .8):
    
    # add perclos column
    feature_df["perclos_closed_combined"] = feature_df["combined_eye_closure"] >= closed_threshold
    
    # add perclos interval
    num_interval_frames = interval_in_sec * fps
    perclos = feature_df["perclos_closed_combined"].to_numpy()
    
    conv_filter = np.ones((num_interval_frames))
    res = np.convolve(perclos, conv_filter, "valid") / num_interval_frames
    res = np.concatenate(([np.NaN] * (len(conv_filter) - 1), res))
    assert perclos.shape == res.shape

    feature_df["perclos_combined"] = res

    return feature_df

def add_blink_features(feature_df: pd.DataFrame, interval_in_sec: int = 60, fps: int = 30):
    num_interval_frames = interval_in_sec * fps 
    perclos = feature_df["perclos_closed_combined"].to_numpy()
    all_blink_properties = np.empty(perclos.shape + (4,))
    all_blink_properties[:] = np.NaN
    
    
    for index in tqdm(list(range(len(perclos)))[num_interval_frames:]):
        condition = perclos[index-num_interval_frames:index]
        if not np.any(condition):
            blink_durations = np.array([0,0,0,0])
        else:
            blink_durations = np.diff(np.where(np.concatenate(([condition[0]],
                                     condition[:-1] != condition[1:],
                                     [True])))[0])[::2]
        all_blink_properties[index, :] = (blink_durations.max(), blink_durations.min(), blink_durations.mean(), len(blink_durations))

    
    feature_df["max_blink_duration_60s_interval"] = all_blink_properties[:,0]
    feature_df["min_blink_duration_60s_interval"] = all_blink_properties[:,1]    
    feature_df["mean_blink_duration_60s_interval"] = all_blink_properties[:,2]    
    feature_df["blink_counts_60s_interval"] = all_blink_properties[:,3]    
    
    return feature_df