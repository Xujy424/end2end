import numpy as np
import bottleneck as bn
import pandas as pd



def corr(a, b, axis):
    b[np.isnan(a)] = np.nan
    a[np.isnan(b)] = np.nan
    arr = (
            (bn.nanmean(a * b, axis=axis) - bn.nanmean(a, axis=axis) * bn.nanmean(b, axis=axis))
            / (bn.nanstd(a, axis=axis) + 1e-6)
            / (bn.nanstd(b, axis=axis) + 1e-6)
    )
    bn.replace(arr, np.nan, 0)
    arr[np.isinf(arr)] = 0
    return arr

def IC(y_, y):
    ics = corr(y_.copy(), y.copy(), axis=1)
    return ics

def rankIC(y_, y):
    rank_ics = corr(bn.nanrankdata(y_.copy(), axis=1), bn.nanrankdata(y.copy(), axis=1), axis=1)
    return rank_ics


def cal_alpha(alpha, label, num_group=10):
    rank = bn.nanrankdata(alpha, axis=1)
    num_signal = np.nanmax(rank, axis=1)
    stock_each_group = num_signal // num_group
    group_ret = np.full((num_group, num_signal.shape[0]), np.nan)
    for i in range(num_group):
        if i==num_group-1:
            group_ix = (rank.T > stock_each_group * i) & (rank.T <= num_signal)
        else:
            group_ix = (rank.T > stock_each_group * i) & (rank.T <= stock_each_group * (i + 1)) # n_stock, n_date
        temp_ret = label.copy()
        temp_ret[~group_ix.T] = np.nan
        group_ret[i] = np.nanmean(temp_ret, axis=1)
    group_ret = group_ret - np.nanmean(group_ret, axis=0)
    col_list = list(range(1, num_group + 1))[::-1]
    group_ret = pd.DataFrame(
        group_ret.T,
        columns=col_list,
        index=alpha.index,
    )
    return group_ret


def cal_sharpe(ret):
    return np.mean(ret) / np.std(ret) * np.sqrt(242)


def cal_maxdrawdown(ret):
    cum = np.cumsum(ret)
    return ((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)).min()


def calc_group_ret(alpha, label, num_group=10):
    rank = bn.nanrankdata(alpha, axis=1)
    num_signal = np.nanmax(rank, axis=1)
    stock_each_group = num_signal // num_group
    group_ret = np.full((num_group, num_signal.shape[0]), np.nan)
    for i in range(num_group):
        group_ix = (rank.T > stock_each_group * i) & (rank.T < stock_each_group * (i + 1))
        temp_ret = label.copy()
        temp_ret[~group_ix.T] = np.nan
        group_ret[i] = np.nanmean(temp_ret, axis=1)
    group_ret = group_ret - np.nanmean(group_ret, axis=0)
    col_list = list(range(1, num_group + 1))[::-1]
    group_ret = pd.DataFrame(
        group_ret.T,
        columns=col_list,
        index=alpha.index,
    )
    group_ret = group_ret.cumsum()
    group_ret.plot(grid=True)
