import pandas as pd


# def get_rolling_windows(start_dt, end_dt, train_len=8, valid_len=2, test_len=1, rolling_gap=1):
#     train_start = pd.to_datetime(start_dt)                                 # 查找第一个大于等于value的索引
#     windows = []
#     while True:
#         valid_start = train_start + pd.DateOffset(years=train_len)
#         train_end = valid_start - pd.Timedelta(days=1)
#         test_start = valid_start + pd.DateOffset(years=valid_len)
#         valid_end = test_start - pd.Timedelta(days=1)
#         test_end = test_start + pd.DateOffset(years=test_len) - pd.Timedelta(days=1)
#         if test_end>pd.to_datetime(end_dt):
#             break
#         windows.append(
#             (((train_start).strftime('%Y-%m-%d'), (train_end).strftime('%Y-%m-%d')),
#              ((valid_start).strftime('%Y-%m-%d'), (valid_end).strftime('%Y-%m-%d')),
#              ((test_start).strftime('%Y-%m-%d'), (test_end).strftime('%Y-%m-%d')))
#         )
#         train_start += pd.DateOffset(years=rolling_gap)   # months,years
#     splitratio = f'{train_len}y{valid_len}y{test_len}y'
#     return windows, splitratio

def get_rolling_windows(start_dt, end_dt, train_len=7, valid_len=1, test_len=1, rolling_gap=1):
    start_dt = pd.to_datetime(start_dt)
    end_dt = pd.to_datetime(end_dt)
    
    train_offset = pd.DateOffset(years=train_len)
    valid_offset = pd.DateOffset(years=valid_len)
    test_offset = pd.DateOffset(years=test_len)
    gap_offset = pd.DateOffset(years=rolling_gap)
    
    windows = []
    train_start = start_dt
    
    while True:
        valid_start = train_start + train_offset
        train_end = valid_start - pd.Timedelta(days=1)
        test_start = valid_start + valid_offset
        valid_end = test_start - pd.Timedelta(days=1)
        test_end = test_start + test_offset - pd.Timedelta(days=1)
        if test_start > end_dt:
            break
        if test_end > end_dt:
            test_end = end_dt
        if test_end < test_start:
            break
        windows.append(
            (
                (train_start.strftime('%Y-%m-%d'), train_end.strftime('%Y-%m-%d')),
                (valid_start.strftime('%Y-%m-%d'), valid_end.strftime('%Y-%m-%d')),
                (test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d'))
            )
        )
        train_start += gap_offset
        if train_start > end_dt:
            break
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