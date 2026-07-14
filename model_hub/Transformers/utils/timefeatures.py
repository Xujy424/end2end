from typing import List
import torch
import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from datetime import datetime



class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])






def time2vec(time_list, freq: str):
    """
    批量时间转模型可用时间特征向量
    :param time_list: 时间字符串列表 e.g. ["2026-05-18 14:30:20", ...]
    :param freq: 频率标识 h/t/s/m/a/w/d/b
    :return: torch.Tensor [seq_len, feat_dim] 归一化特征
    """
    feat_list = []
    for t_str in time_list:
        dt = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
        
        # 基础原始值
        month = dt.month
        day = dt.day
        hour = dt.hour
        minute = dt.minute
        second = dt.second
        weekday = dt.weekday()       # 0=周一,6=周日
        year = dt.year
        
        # 归一化到 0~1
        norm_month = (month - 1) / 11
        norm_day = (day - 1) / 30
        norm_hour = hour / 23
        norm_min = minute / 59
        norm_sec = second / 59
        norm_weekday = weekday / 6
        
        # 年内周数、周内天数
        week_of_year = dt.isocalendar()[1]
        norm_week = week_of_year / 52
        norm_week_day = (weekday + 1) / 7
        
        # 年份偏移(以2020为基准)
        year_off = (year - 2020) / 10

        feat = []
        if freq == 'h':
            # 4维 [月,日,时,星期]
            feat = [norm_month, norm_day, norm_hour, norm_weekday]
        elif freq == 't':
            # 5维 [月,日,时,分,星期]
            feat = [norm_month, norm_day, norm_hour, norm_min, norm_weekday]
        elif freq == 's':
            # 6维 [月,日,时,分,秒,星期]
            feat = [norm_month, norm_day, norm_hour, norm_min, norm_sec, norm_weekday]
        elif freq == 'm':
            # 1维 归一化月份
            feat = [norm_month]
        elif freq == 'a':
            # 1维 年份偏移
            feat = [year_off]
        elif freq == 'w':
            # 2维 [年内周数,周内天数]
            feat = [norm_week, norm_week_day]
        elif freq in ['d', 'b']:
            # 3维 [月,日,星期] 日级/工作日通用
            feat = [norm_month, norm_day, norm_weekday]
        
        feat_list.append(feat)
    
    return torch.tensor(np.array(feat_list), dtype=torch.float32)




def generate_temporal_embedding_input(time_list, freq: str):
    """
    为 TemporalEmbedding 生成符合要求的输入 x（整数型，未归一化）
    输入 x 形状：[batch_size, seq_len, 5]
    5 维固定顺序：[月, 日, 星期几, 小时, 分钟]
    """
    features = []
    
    for time_str in time_list:
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        
        month = dt.month                # 1~12
        day = dt.day                    # 1~31
        weekday = dt.weekday() + 1      # 1~7（1=周一，7=周日）
        hour = dt.hour                  # 0~23
        minute = dt.minute              # 0~59
        
        # 固定 5 维顺序 [月, 日, 星期, 小时, 分钟]
        feat = [month, day, weekday, hour, minute]
        features.append(feat)
    
    # 转成张量
    x = torch.tensor(features, dtype=torch.long)
    
    # 一般模型需要 [seq_len, 5] → 加 batch 变成 [1, seq_len, 5]
    x = x.unsqueeze(0)
    return x