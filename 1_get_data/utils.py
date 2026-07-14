import numpy as np
import bottleneck as bn
from abc import ABC, abstractmethod
from numpy.lib.stride_tricks import sliding_window_view
import os
from typing import Dict
import pandas as pd
from sklearn.preprocessing import PowerTransformer, RobustScaler
from scipy.stats import rankdata
from pathlib import Path


class Calculator(ABC):

    @staticmethod
    def rolling_retprod(x: np.ndarray, window: int, future: bool = True) -> np.ndarray:
        ret_factor = 1 + x  # 转为累积因子
        T, N = x.shape
        if future:
            view = np.lib.stride_tricks.sliding_window_view(ret_factor, window, axis=0)
            cumprod = np.nanprod(view, axis=-1) # 取每个窗口最后一个累积乘积
            res = cumprod - 1
            pad = np.full((window-1, N), np.nan)
            return np.concatenate([res, pad], axis=0)
        else:
            pad = ((window-1, 0), (0, 0))
            x_pad = np.pad(ret_factor, pad, mode='constant', constant_values=1)
            view = np.lib.stride_tricks.sliding_window_view(x_pad, window, axis=0)
            cumprod = np.nanprod(view, axis=-1)
            return cumprod - 1

    @staticmethod
    def rolling_nanmean(x: np.ndarray, window: int) -> np.ndarray:
        """滚动 nanmean """
        pad = ((window - 1, 0), (0, 0))
        x_pad = np.pad(x, pad, mode='constant', constant_values=np.nan)
        windows = np.lib.stride_tricks.sliding_window_view(x_pad, window, axis=0)
        return np.nanmean(windows, axis=-1)

    @staticmethod
    def rolling_nanstd(x: np.ndarray, window: int, ddof: int = 0) -> np.ndarray:
        """滚动 nanstd """
        pad = ((window - 1, 0), (0, 0))
        x_pad = np.pad(x, pad, mode='constant', constant_values=np.nan)
        windows = np.lib.stride_tricks.sliding_window_view(x_pad, window, axis=0)
        return np.nanstd(windows, axis=-1, ddof=ddof)

    @staticmethod
    def rolling_nancorr(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
        """滚动 Pearson 相关系数（支持 NaN）"""
        out = np.full_like(x, np.nan)
        xw = np.lib.stride_tricks.sliding_window_view(x, window, axis=0)
        yw = np.lib.stride_tricks.sliding_window_view(y, window, axis=0)

        mx = np.nanmean(xw, axis=-1, keepdims=True)
        my = np.nanmean(yw, axis=-1, keepdims=True)

        cov = np.nanmean((xw - mx) * (yw - my), axis=-1)
        sx = np.nanstd(xw, axis=-1)
        sy = np.nanstd(yw, axis=-1)

        cor = np.divide(cov, sx*sy, out=np.full_like(cov, 0), where=(sx*sy)!=0)

        out[window - 1:] = cor
        return out

    @staticmethod
    def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """安全除法，避免除0与nan爆炸"""
        res = np.divide(a, b, out=np.full_like(a, 0), where=b!=0)
        return res

    @staticmethod
    def rolling_ema(x: np.ndarray, window: int) -> np.ndarray:
        alpha = 2.0 / (window + 1.0)
        ar = np.arange(x.shape[0], dtype=np.float32)
        weights = (1.0 - alpha) ** ar
        weights = weights.reshape(-1, 1)

        mask = ~np.isnan(x)
        x_filled = np.where(mask, x, 0.0)

        w_sum = np.cumsum(x_filled * weights, axis=0)
        mask_sum = np.cumsum(mask * weights, axis=0)
        ema = w_sum / mask_sum

        ema[mask_sum < 1e-8] = np.nan
        return ema

    @staticmethod
    def rolling_max(x: np.ndarray, window: int) -> np.ndarray:
        """MAX 滚动最大值 | 全向量化"""
        x_pad = np.pad(x, ((window - 1, 0), (0, 0)), constant_values=np.nan)
        windows = np.lib.stride_tricks.sliding_window_view(x_pad, window, axis=0)
        return np.nanmax(windows, axis=-1)

    @staticmethod
    def rolling_min(x: np.ndarray, window: int) -> np.ndarray:
        """MIN 滚动最小值 | 全向量化"""
        x_pad = np.pad(x, ((window - 1, 0), (0, 0)), constant_values=np.nan)
        windows = np.lib.stride_tricks.sliding_window_view(x_pad, window, axis=0)
        return np.nanmin(windows, axis=-1)
    
    @staticmethod
    def diff(x: np.ndarray, periods: int = 1) -> np.ndarray:
        res = np.full_like(x, np.nan)
        res[periods:] = x[periods:] - x[:-periods]
        return res
    
    @staticmethod
    def pct_change(x: np.ndarray, periods: int = 1) -> np.ndarray:
        res = np.full(x.shape, np.nan)

        current = x[periods:]
        base = x[:-periods]

        x_diff = current - base

        valid = ~np.isnan(current) & ~np.isnan(base)
        nonzero_base = valid & (base != 0)
        zero_base = valid & (base == 0)

        res_part = np.full(x_diff.shape, np.nan)

        np.divide(
            x_diff,
            base,
            out=res_part,
            where=nonzero_base,
        )

        res_part[zero_base] = 0.0
        res[periods:] = res_part
        return res




class Processor:

    @staticmethod
    def calc_indmv_neutral_longshort(ind_signal, temp_mv):
        ix = ~(np.isnan(ind_signal) | np.isinf(ind_signal) | np.isnan(temp_mv) | np.isinf(temp_mv))
        ind_signal[~ix] = np.nan
        temp_mv[~ix] = np.nan

        mv_mean = bn.nanmean(temp_mv, axis=1)
        signal_mean = bn.nanmean(ind_signal, axis=1)
        m = (mv_mean * signal_mean - bn.nanmean(temp_mv * ind_signal, axis=1)) / (mv_mean**2 - bn.nanmean(temp_mv**2, axis=1) + 1e-6)
        b = signal_mean - m * mv_mean
        residual = (ind_signal.T - (temp_mv.T * m) - b).T
        ind_signal = (residual.T - bn.nanmean(residual, axis=1)) / (bn.nanstd(residual, axis=1) + 1e-6)
        return ind_signal.T

    @staticmethod
    def indmv_neutral_longshort(alpha_vec, ind_arr, mv_arr):
        new_signal = np.full_like(alpha_vec, np.nan)   # [T,N]
        ln_mv = np.log(mv_arr)
        for i in range(31):
            ind_ix = ind_arr == i
            ind_select = ind_ix.any(axis=0)
            ind_ix_select = ind_ix[:, ind_select]
            ind_signal = alpha_vec[:, ind_select].copy()
            ind_signal[~ind_ix_select] = np.nan
            temp_mv = ln_mv[:, ind_select].copy()
            new_signal[ind_ix] = Processor.calc_indmv_neutral_longshort(ind_signal, temp_mv)[ind_ix_select]
        return new_signal

    @staticmethod
    def cross_standardize(x):
        mean = np.nanmean(x, axis=1, keepdims=True)
        std = np.nanstd(x, axis=1, keepdims=True)
        std = np.where((std < 1e-8) | np.isnan(std), np.nan, std)
        x = np.divide(x-mean, std, out=np.full_like(x, 1), where=std!=0)
        return x

    @staticmethod
    def series_standardize( feat: np.ndarray, l: int) -> np.ndarray:
        T, N = feat.shape
        windows = sliding_window_view(feat, l, axis=0)
        denominator = windows[:, :, :1]  # (T-l+1, N, 1)
        ratios = np.divide(windows, denominator, out=np.full_like(windows, 0), where=denominator!=0)
        out = np.full((T, N, l), np.nan, dtype=feat.dtype)
        out[l - 1:, :, :] = ratios  # 从第 l-1 行开始放置有效数据
        return out
    
    @staticmethod
    def winsorize_clip(arr:np.ndarray, p=0.01, n=3, type='mad') -> np.ndarray:
        if type=='quantile':
            lower = np.nanquantile(arr, p, axis=1, keepdims=True)
            upper = np.nanquantile(arr, 1-p, axis=1, keepdims=True)
        elif type=='sigma':
            mean = np.nanmean(arr, axis=1, keepdims=True)
            std = np.nanstd(arr, axis=1, keepdims=True)
            lower = mean - n * std
            upper = mean + n * std
        elif type=='mad':
            median = np.nanmedian(arr, axis=1, keepdims=True)
            mad = np.nanmedian(np.abs(arr-median), axis=1, keepdims=True)
            lower = median - n * 1.4826 * mad
            upper = median + n * 1.4826 * mad
        clipped = np.clip(arr, lower, upper)
        return clipped
    
    @staticmethod
    def winsorize_midtrim(arr: np.ndarray, p=0.01) -> np.ndarray:
        l = np.nanquantile(arr, p, axis=1, keepdims=True)
        u = np.nanquantile(arr, 1-p, axis=1, keepdims=True)
        low_mask = (arr < l) & (~np.isnan(arr))
        low_mean = np.nanmean(np.where(low_mask, arr, np.nan), axis=1, keepdims=True)
        high_mask = (arr > u) & (~np.isnan(arr))
        high_mean = np.nanmean(np.where(high_mask, arr, np.nan), axis=1, keepdims=True)
        res = np.where(low_mask, low_mean, arr)
        res = np.where(high_mask, high_mean, res)
        return res
    
    @staticmethod
    def winsorize_linearsmooth(arr: np.ndarray, p=0.01, shrink=0.5):
        res = arr.copy()
        lower = np.nanquantile(res, p, axis=1, keepdims=True)
        upper = np.nanquantile(res, 1-p, axis=1, keepdims=True)
        mask_low = (res < lower) & (~np.isnan(res))
        mask_high = (res > upper) & (~np.isnan(res))
        res = np.where(mask_low, lower-(lower-res)*shrink, res)
        res = np.where(mask_high, upper+(res-upper)*shrink, res)
        return res
    
    @staticmethod
    def rank_transform(x: np.ndarray, normalize=True) -> np.ndarray:
        x = x.copy()
        mask = np.isnan(x).all(axis=1)
        arr = x[~mask]
        ranks = rankdata(arr, axis=1, method='average', nan_policy='omit')
        ranks = ranks - 1
        if normalize:
            valid_counts = np.sum(~np.isnan(arr), axis=1, keepdims=True)
            denominator = valid_counts - 1
            denominator = np.where(denominator > 0, denominator, 1)
            ranks = ranks / denominator
        x[~mask] = ranks
        return x

    @staticmethod
    def yeojohnson(x: np.ndarray) -> np.ndarray:
        pt = PowerTransformer(method='yeo-johnson')
        x = x.copy()
        mask = np.isnan(x).all(axis=1)
        arr = x[~mask]
        res = pt.fit_transform(arr.T).T
        x[~mask] = res
        return x
    
    @staticmethod
    def robustscaler(arr: np.ndarray) -> np.ndarray:
        rs = RobustScaler()
        x = x.copy()
        mask = np.isnan(x).all(axis=1)
        arr = x[~mask]
        res = rs.fit_transform(arr.T).T
        x[~mask] = res
        return x
    


class Loader:
    def __init__(self, data_dir):
        self.root = Path(data_dir)
        self.dates = np.load(self.root/'axis'/'dates.npy', allow_pickle=True)     
        self.ticks = np.load(self.root/'axis'/'ticks.npy', allow_pickle=True) 
        self.T = len(self.dates)
        self.N = len(self.ticks)

    def load_daily_feats(self) -> Dict[str, np.ndarray]:
        features = {}
        for feat_name in ["open_adj", "close_adj", "high_adj", "low_adj", "volume_adj", "amount", "turnover", 'logmv']:
            file_path = self.root/"d_field"/f"{feat_name}.bin"
            features[feat_name] = np.memmap(file_path, shape=(self.T, self.N), dtype=float)  # 大文件用内存映射
        return features
    
    def load_min_feats(self) -> Dict[str, np.ndarray]:
        features = {}
        for feat_name in ["open_adj", "close_adj", "high_adj", "low_adj", "volume_adj", "amount"]:
            file_path = self.root/"m_field"/f"{feat_name}.bin"
            features[feat_name] = np.memmap(file_path, shape=(self.T, self.N, 241), dtype=float)  # 大文件用内存映射
        return features

    def load_mask(self) -> Dict[str, np.ndarray]:
        masks = {}
        for name in ['industry','sector','tradable','hs300_mask','zz500_mask','a500_mask','zz1000_mask','zz2000_mask']:
            file_path = self.root/"mask"/f"{name}.bin"
            if name in ['industry','sector']:
                masks[name] = np.memmap(file_path, shape=(self.T, self.N), dtype=float, mode='r')
            else:
                masks[name] = np.memmap(file_path, shape=(self.T, self.N), dtype=bool, mode='r')  # 大文件用内存映射
        return masks
    
    def load_barra(self) -> Dict[str, np.ndarray]:
        feats = {}
        for name in ['beta','btop','size','nonlinear_size','momentom','residual_vol','liquidity','leverage','growth1','earnings_yield']:
            file_path = self.root/"barra"/f"{name}.bin"
            feats[name] = np.memmap(file_path, shape=(self.T, self.N), dtype=float)  # 大文件用内存映射
        return feats


