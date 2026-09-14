from pathlib import Path
import numpy as np

from matrix_math import *

ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else Path('/data/shanghai/xujiayi/workflow/data/')
label_dir = ROOT/"stock/model_input/labels/"

dates = np.load(ROOT/"axis/dates.npy", allow_pickle=True)
stock_ticks = np.load(ROOT/"axis/stock_ticks.npy", allow_pickle=True)
pct = np.memmap(
    ROOT/"stock/d_essentials/pct.bin", 
    dtype="float32", mode="r", shape=(len(dates), len(stock_ticks))
)

def calc_forward_return(pct, offset=2, horizon=5):
    '''计算不同周期的未来收益率'''
    windows = np.lib.stride_tricks.sliding_window_view(
        pct[offset:], horizon, axis=0
    )
    forward_return = np.prod(1+windows, axis=-1)-1
    label = np.full(pct.shape, np.nan, dtype=np.float32)
    label[:len(forward_return)] = forward_return
    label.astype(np.float32).tofile(label_dir/f"Y.{horizon}D.bin")

def calc_zscore_return(h=1):
    '''计算未来收益率的zscore'''
    label = np.memmap(
        label_dir/f"Y.{h}D.bin", dtype="float32", mode="r", shape=(len(dates), len(stock_ticks))
    )
    label = winsorize(label)
    label = cross_sectional_zscore(label)
    label.astype(np.float32).tofile(label_dir/f"Y.{h}D.zscore.bin")

def calc_neutral_return(h=1):
    '''计算中性化后的未来收益率'''
    label = np.memmap(
        label_dir/f"Y.{h}D.bin", dtype="float32", mode="r", shape=(len(dates), len(stock_ticks))
    )
    industry_mask = np.memmap(
        ROOT/"stock/industry/industry.bin", dtype="float32", mode="r", shape=(len(dates), len(stock_ticks))
    )
    mv = np.memmap(
        ROOT/"stock/d_essentials/circ_mv.bin", dtype="float32", mode="r", shape=(len(dates), len(stock_ticks))
    )
    neutral_label = winsorize(label)
    neutral_label = industry_size_neutralize(neutral_label, industry_mask, mv)
    neutral_label = cross_sectional_zscore(neutral_label)
    neutral_label.astype(np.float32).tofile(label_dir/f"Y.{h}D.neutral.bin")

def calc_zcorr_return(pct, w=1, topk=50, eps=1e-6):
    '''计算相关性聚类内标准化的未来收益率标签'''
    T,N = pct.shape[0], pct.shape[1]
    y10 = calc_forward_return(pct, offset=2, horizon=10)
    past_view = np.lib.stride_tricks.sliding_window_view(
        pct,
        window_shape=w,
        axis=0,
    ) 
    top50_idx = np.full((T, N, topk), np.nan, dtype=np.int64)
    peer_mean_arr = np.full((T, N), np.nan, dtype=np.float32)
    peer_std_arr = np.full((T, N), np.nan, dtype=np.float32)
    zcorr_label = np.full((T, N), np.nan, dtype=np.float32)
    for k, x in enumerate(past_view):
        t = k + w - 1
        x = np.asarray(x, dtype=np.float32)
        # 过去10日数据完整的股票
        valid = np.isfinite(x).all(axis=1)
        # 有效股票不足51只，无法为每只股票选择50只其他股票
        if valid.sum() <= topk:
            continue
        # 无效行填0，避免NaN参与矩阵运算
        x = np.where(valid[:, None], x, 0.0)
        # 去均值并单位化
        x -= x.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(x, axis=1)
        valid &= norm > eps
        x = np.divide(
            x,
            norm[:, None],
            out=np.zeros_like(x),
            where=norm[:, None] > eps,
        )
        # Pearson相关系数矩阵
        corr = x @ x.T
        # 无效股票不能作为候选股票
        corr[:, ~valid] = -np.inf
        # 排除自身
        np.fill_diagonal(corr, -np.inf)
        # 每行选择相关性最大的50只股票，不进行内部排序
        idx = np.argpartition(
            corr,
            kth=N - topk,
            axis=1,
        )[:, -topk:]

        # 只有目标股票自身有效时，才保存Top50
        top50_idx[t, valid] = idx[valid]
        # 取Top50股票当日的未来10日收益
        peer_y10 = y10[t, idx]  # [N, topk]
        # 自动忽略Top50中y10为nan的股票
        peer_mean = np.nanmean(peer_y10, axis=1)
        peer_mean_arr[t] = peer_mean
        peer_std = np.nanstd(peer_y10, axis=1)
        peer_std_arr[t] = peer_std
        # 用Top50股票的均值、标准差标准化自己的y10
        can_calc = (
            valid
            & np.isfinite(y10[t])
            & np.isfinite(peer_mean)
            & (peer_std > eps)
        )
        zcorr_label[t] = np.divide(
            y10[t] - peer_mean,
            peer_std,
            out=np.full(N, np.nan, dtype=np.float32),
            where=can_calc,
        )
    zcorr_label.astype(np.float32).tofile(label_dir/f"Y.{w}D.zcorr.bin")


def run_label(pct, offset=2, horizon=[1,5,10,20]):
    for h in horizon:
        calc_forward_return(pct, offset=offset, horizon=h)
        calc_zscore_return(h=h)
        calc_neutral_return(h=h)
        calc_zcorr_return(pct, w=10, topk=50, eps=1e-6)


if __name__ == "__main__":
    run_label(pct, offset=2, horizon=[1,5,10,20])
