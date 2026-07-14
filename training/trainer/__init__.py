import pandas as pd


def get_rolling_windows(start_dt, end_dt, train_len=8, valid_len=2, test_len=1, rolling_gap=1):
    train_start = pd.to_datetime(start_dt)                                 # 查找第一个大于等于value的索引
    windows = []
    while True:
        valid_start = train_start + pd.DateOffset(years=train_len)
        train_end = valid_start - pd.Timedelta(days=1)
        test_start = valid_start + pd.DateOffset(years=valid_len)
        valid_end = test_start - pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(years=test_len) - pd.Timedelta(days=1)
        if test_end>pd.to_datetime(end_dt):
            break
        windows.append(
            (((train_start).strftime('%Y-%m-%d'), (train_end).strftime('%Y-%m-%d')),
             ((valid_start).strftime('%Y-%m-%d'), (valid_end).strftime('%Y-%m-%d')),
             ((test_start).strftime('%Y-%m-%d'), (test_end).strftime('%Y-%m-%d')))
        )
        train_start += pd.DateOffset(years=rolling_gap)   # months,years
    splitratio = f'{train_len}y{valid_len}y{test_len}y'
    return windows, splitratio


from .basic_supervise import BasicSuperviseTrainer
from .rolling_supervise import RollingSuperviseTrainer
from .basic_selfsupervise import BasicSelfSuperviseTrainer
from .rolling_selfsupervise import RollingSelfSuperviseTrainer

TRAINER_DICT = {
    'basic_supervise': BasicSuperviseTrainer,
    'rolling_supervise': RollingSuperviseTrainer,
    'basic_selfsupervise': BasicSelfSuperviseTrainer,
    'rolling_selfsupervise': RollingSelfSuperviseTrainer,
}