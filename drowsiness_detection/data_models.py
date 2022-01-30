from functools import cached_property
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd


class PreComputedSlicer:
    def __init__(self, indices: List[int], object_dict: Dict[int, object], index_interval: int):
        indices_dict = {}
        for index in indices:
            objects = []
            for key, object in object_dict.items():
                if key < index and (index - key) <= index_interval:
                    objects.append(object)
                if key > index:
                    break
            indices_dict[index] = objects
        self.indices_dict = indices_dict

    def __getitem__(self, item):
        return self.indices_dict[item]


class BlinkEvent:

    def __init__(self, indices: List[int], data: pd.DataFrame, reopening_threshold: float):
        """indices contains the index of each value of one blink event. The first
        marks the start and the last one the end."""
        self.reopening_threshold = reopening_threshold
        self.data = data
        self.indices = indices
        self.event_data: pd.DataFrame = self.data.iloc[self.indices]
        self._reopen_idx = None
        self._closing_end_idx = None
        self._blink_interval = -1
        self._max_closing_speed_idx = None

    @cached_property
    def full_closure_idx(self):
        return self.event_data.idxmax()

    @cached_property
    def full_closure(self):
        return self.event_data[self.full_closure_idx]

    @cached_property
    def amplitude(self):
        return (self.event_data.max() - self.event_data.min())  # [0]

    @cached_property
    def start_idx(self):
        return self.indices[0]

    @cached_property
    def start(self):
        return self.event_data[self.start_idx]

    @cached_property
    def end_idx(self):
        return self.indices[-1]

    @cached_property
    def end(self):
        return self.event_data[self.end_idx]

    @cached_property
    def reopen_start_idx(self):
        if self._reopen_idx:
            return self._reopen_idx
        reopen_idx = self.full_closure_idx
        try:
            for idx in self.indices[self.indices.index(reopen_idx + 1):]:
                if abs(self.event_data[reopen_idx] - self.event_data[idx]) < self.reopening_threshold:
                    reopen_idx = idx
                else:
                    break
            self._reopen_idx = reopen_idx
        except ValueError as e:
            self._reopen_idx = reopen_idx

        closing_end_idx = self.full_closure_idx
        try:
            for idx in self.indices[self.indices.index(reopen_idx - 1)::-1]:
                if abs(self.event_data[closing_end_idx] - self.event_data[idx]) < self.reopening_threshold:
                    closing_end_idx = idx
                else:
                    break
            self._closing_end_idx = closing_end_idx
        except ValueError as e:
            self._closing_end_idx = closing_end_idx
        return self._reopen_idx

    @cached_property
    def closing_end_idx(self):
        if self._closing_end_idx:
            return self._closing_end_idx
        _ = self.reopen_start_idx
        return self._closing_end_idx

    @cached_property
    def blink_interval(self):
        if self._blink_interval == -1:
            raise ValueError("blink interval can only be set externally.")
        return self._blink_interval

    @cached_property
    def closing_speed(self):
        time = self.closing_end_idx - self.start_idx
        if time == 0:
            return 0
        return self.amplitude / time

    @cached_property
    def max_closing_speed(self):
        closing_data = self.event_data[self.start_idx:self.closing_end_idx]
        if len(closing_data) < 2:
            self._max_closing_speed_idx = self.closing_end_idx
            return 0
        else:
            z = closing_data.diff()
            self._max_closing_speed_idx = z[z == z.max()].index[0]
            return z.max()

    @cached_property
    def max_closing_speed_idx(self):
        if self._max_closing_speed_idx:
            return self._max_closing_speed_idx
        _ = self.max_closing_speed
        return self._max_closing_speed_idx

    @cached_property
    def blink_duration(self):
        return self.max_closing_speed_idx - self.start_idx

    @cached_property
    def lid_opening_delay(self):
        return self.reopen_start_idx - self.closing_end_idx

    def plot(self, num_extra_frames: int = 60):
        fig, ax = plt.subplots()
        ax.plot(self.data[max(0, self.start_idx - num_extra_frames): min(len(self.data), self.end_idx + num_extra_frames)])
        ax.plot([self.start_idx, self.end_idx], [self.data.loc[self.start_idx, 0], self.data.loc[self.end_idx, 0]], "ro", ms=5)
        plt.show()
