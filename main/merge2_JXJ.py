import numpy as np
from sklearn.preprocessing import PowerTransformer, RobustScaler
from scipy.stats import rankdata
import pandas as pd

# test_alpha = pd.read_pickle('/data/shanghai/data/backtest/test_port.pkl')
# print(test_alpha)

alpha = pd.read_csv("0_result/gru/rolling/alpha_merge_20210104_20251231.csv",index_col=0)
alpha.columns = [col + ('.SZ' if col[0] in ('0','3') else '.SH') for col in alpha.columns]
alpha.index.name = 'trade_dt'
alpha.columns.name = 'stcode'
alpha = pd.DataFrame(alpha.stack(),columns=['Predict'])
alpha.dropna(inplace=True)
print(alpha)
alpha.to_pickle('/data/data/alpha/xjy_rolling_minute_gru.pkl')

# dates = np.load('/data/xujiayi/end2end/axis/dates.npy', allow_pickle=True)     
# ticks = np.load('/data/xujiayi/end2end/axis/ticks.npy', allow_pickle=True) 
# close = np.memmap('/data/xujiayi/end2end/GRU_new/close_adj.bin', shape=(len(dates),len(ticks)), dtype=float, mode='r')
# print(close.shape)


# alpha = pd.read_csv("0_result/gru/rolling/alpha_merge_20210104_20251231.csv",index_col=0)
# def yeojohnson(x: np.ndarray) -> np.ndarray:
#         pt = PowerTransformer(method='yeo-johnson')
#         x = x.copy()
#         mask = np.isnan(x).all(axis=1)
#         arr = x[~mask]
#         res = pt.fit_transform(arr.T).T
#         x[~mask] = res
#         return x
# arr = yeojohnson(alpha.values)
# df = pd.DataFrame(arr, index=alpha.index, columns=alpha.columns)
# alpha.to_csv("0_result/gru/rolling/mad_alpha.csv")