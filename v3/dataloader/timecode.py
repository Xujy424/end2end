from __future__ import annotations

import numpy as np
import pandas as pd

from .vanilla import BaseDataset


class DailyTimeCode:
    def __init__(self, dates: pd.DatetimeIndex, lag):
        self.time_feat = np.stack(
            [
                dates.month.values.astype(np.int32),
                dates.day.values.astype(np.int32),
                dates.weekday.values.astype(np.int32),
            ],
            axis=1,
        )
        self.lag = lag

    def get_time_seq(self, idx):
        return self.time_feat[idx - self.lag + 1: idx + 1]


class MinuteTimeCode:
    def __init__(self, dates: pd.DatetimeIndex):
        self.dates = dates
        self.total_len = 241
        morning = np.arange(9 * 60 + 30, 11 * 60 + 30 + 1)
        afternoon = np.arange(13 * 60 + 1, 15 * 60 + 1)
        total_mins = np.concatenate([morning, afternoon])
        self.hour = total_mins // 60
        self.minute = total_mins % 60

    def get_time_seq(self, idx):
        dt = self.dates[idx]
        month_arr = np.full(self.total_len, dt.month, dtype=np.int32)
        day_arr = np.full(self.total_len, dt.day, dtype=np.int32)
        weekday_arr = np.full(self.total_len, dt.weekday(), dtype=np.int32)
        feat = np.stack(
            [month_arr, day_arr, weekday_arr, self.hour.astype(np.int32), self.minute.astype(np.int32)],
            axis=1,
        )
        return feat[1:-3]


class TimeCodeBase(BaseDataset):
    def __init__(self, start_date, end_date, freq, lag=None):
        super().__init__(start_date=start_date, end_date=end_date)
        dates_dt = pd.to_datetime(self.dates)
        if freq == "daily":
            self.timecode = DailyTimeCode(dates_dt, lag)
            self.lag = lag
        elif freq == "intraday":
            self.timecode = MinuteTimeCode(dates_dt)
        else:
            raise ValueError(f"Unsupported time freq: {freq}")

    def _get_daily_feat(self, date_idx, tick_indices):
        timecode_seq = self.timecode.get_time_seq(date_idx)
        return np.tile(timecode_seq[np.newaxis, :, :], (len(tick_indices), 1, 1))

    def _load_fields(self):
        pass

    def _load_labels(self):
        pass

    def _init_dataset(self):
        pass

    def _get_label(self, date_idx, tick_indices):
        pass
