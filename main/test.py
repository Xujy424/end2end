#from ..training.metrics import IC, rankIC, calc_group_ret
import bottleneck as bn
import bisect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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

    
data_path = "0_result/gru/rolling/"
pred = pd.read_csv("/home/xujiayi/PycharmProjects/Models/XJY_end2end/0_result/gru/rolling/alpha_merge_20210104_20251231.csv", index_col=0)
dates = np.load('/data/xujiayi/end2end/axis/dates.npy', allow_pickle=True)
ticks = np.load('/data/xujiayi/end2end/axis/ticks.npy', allow_pickle=True)
label = np.memmap('/data/xujiayi/end2end/label/Y.10D.bin', dtype=float, mode='r', shape=(len(dates), len(ticks))) 
label = label[bisect.bisect_left(dates, pd.to_datetime('2021-01-01')):bisect.bisect_right(dates, pd.to_datetime('2025-12-31'))]
# pred = pred.loc[pred.index>='2020-01-01']

print(f'mean_RankIC:{np.mean(rankIC(pred,label)):.3%},  mean_IC:{np.mean(IC(pred,label)):.3%}')

calc_group_ret(pred, label)
plt.title(f'mean_RankIC:{np.mean(rankIC(pred,label)):.3%},  mean_IC:{np.mean(IC(pred,label)):.3%}')
plt.savefig(data_path+'GroupRet_rawY.png')
plt.show()
rankics, ics = rankIC(pred, label), IC(pred, label)
plt.figure(figsize=(10, 6))
plt.plot(pd.to_datetime(pred.index), np.cumsum(rankics), label='test_rankics')
plt.plot(pd.to_datetime(pred.index), np.cumsum(ics), label='test_ics')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=30, ha="right")
plt.legend()
plt.savefig(data_path+'cumsumIC_rawY.png')
plt.show()