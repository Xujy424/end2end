from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matrix_math import *
from v3.dataset.datapool import ROOT

label_dir = ROOT / "stock/model_input/labels/"
label_dir.mkdir(parents=True, exist_ok=True)

dates = np.load(ROOT / "axis/dates.npy", allow_pickle=True)
stock_ticks = np.load(ROOT / "axis/stock_ticks.npy", allow_pickle=True)
pct = np.memmap(
    ROOT / "stock/d_essentials/pct.bin",
    dtype="float32",
    mode="r",
    shape=(len(dates), len(stock_ticks)),
)


def calc_forward_return(pct, offset=2, horizon=5):
    """Calculate forward compounded return labels."""
    windows = np.lib.stride_tricks.sliding_window_view(pct[offset:], horizon, axis=0)
    forward_return = np.prod(1 + windows, axis=-1) - 1
    label = np.full(pct.shape, np.nan, dtype=np.float32)
    label[: len(forward_return)] = forward_return
    label.astype(np.float32).tofile(label_dir / f"Y.{horizon}D.bin")
    return label


def _load_forward_label(horizon):
    return np.memmap(
        label_dir / f"Y.{horizon}D.bin",
        dtype="float32",
        mode="r",
        shape=(len(dates), len(stock_ticks)),
    )


def calc_zscore_return(horizon=1):
    """Cross-sectional winsorized z-score of forward returns."""
    label = winsorize(_load_forward_label(horizon))
    label = cross_sectional_zscore(label)
    label.astype(np.float32).tofile(label_dir / f"Y.{horizon}D.zscore.bin")
    return label


def calc_neutral_return(horizon=1):
    """Industry and size neutralized z-score of forward returns."""
    label = _load_forward_label(horizon)
    industry_mask = np.memmap(
        ROOT / "stock/industry/industry.bin",
        dtype="float32",
        mode="r",
        shape=(len(dates), len(stock_ticks)),
    )
    mv = np.memmap(
        ROOT / "stock/d_essentials/circ_mv.bin",
        dtype="float32",
        mode="r",
        shape=(len(dates), len(stock_ticks)),
    )
    neutral_label = winsorize(label)
    neutral_label = industry_size_neutralize(neutral_label, industry_mask, mv)
    neutral_label = cross_sectional_zscore(neutral_label)
    neutral_label.astype(np.float32).tofile(label_dir / f"Y.{horizon}D.neutral.bin")
    return neutral_label


def calc_zcorr_return(pct, horizon=10, lookback=10, topk=50, eps=1e-6):
    """Peer-normalized forward return inside rolling correlation clusters."""
    label_path = label_dir / f"Y.{horizon}D.bin"
    y = _load_forward_label(horizon) if label_path.exists() else calc_forward_return(pct, horizon=horizon)
    T, N = pct.shape
    past_view = np.lib.stride_tricks.sliding_window_view(pct, window_shape=lookback, axis=0)
    zcorr_label = np.full((T, N), np.nan, dtype=np.float32)

    for k, x in enumerate(past_view):
        t = k + lookback - 1
        x = np.asarray(x, dtype=np.float32)
        valid = np.isfinite(x).all(axis=1)
        if valid.sum() <= topk:
            continue

        x = np.where(valid[:, None], x, 0.0)
        x -= x.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(x, axis=1)
        valid &= norm > eps
        x = np.divide(x, norm[:, None], out=np.zeros_like(x), where=norm[:, None] > eps)

        corr = x @ x.T
        corr[:, ~valid] = -np.inf
        np.fill_diagonal(corr, -np.inf)
        idx = np.argpartition(corr, kth=N - topk, axis=1)[:, -topk:]

        peer_y = y[t, idx]
        peer_mean = np.nanmean(peer_y, axis=1)
        peer_std = np.nanstd(peer_y, axis=1)
        can_calc = valid & np.isfinite(y[t]) & np.isfinite(peer_mean) & (peer_std > eps)
        zcorr_label[t] = np.divide(
            y[t] - peer_mean,
            peer_std,
            out=np.full(N, np.nan, dtype=np.float32),
            where=can_calc,
        )

    zcorr_label.astype(np.float32).tofile(label_dir / f"Y.{horizon}D.zcorr.bin")
    return zcorr_label


def run_label(pct, offset=2, horizon=(1, 5, 10, 20), corr_lookback=10, topk=50):
    for h in horizon:
        calc_forward_return(pct, offset=offset, horizon=h)
        calc_zscore_return(horizon=h)
        calc_neutral_return(horizon=h)
        calc_zcorr_return(pct, horizon=h, lookback=corr_lookback, topk=topk, eps=1e-6)


if __name__ == "__main__":
    run_label(pct, offset=2, horizon=(1, 5, 10, 20))
