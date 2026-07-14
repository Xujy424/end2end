# -*- coding: utf-8 -*-
"""
Build PIT financial-report event data for ered_event_model.py.

This script replaces the old notebook logic that pivoted features only on
sheet_infodate and filled other days with zero. That old output is sparse and
announcement-day-only; the model then has almost no event signal on ordinary
trading days.

Correct design
--------------
1. Keep reports in long format: one row = one PIT-visible financial event.
2. Do NOT ffill a daily financial-feature matrix.
3. Do NOT train the model directly on announcement-day-only pivot matrices.
4. Save an event store sorted by (tick_idx, effective_idx).
5. In the Dataset, for any sample (trade_date, stock), retrieve the last K
   events with effective_idx <= trade_idx and compute event_age [L, K].

The model consumes:
    market_x:   [L, Dm]
    event_x:    [K, De]
    event_mask: [K]
    event_age:  [L, K]

where event_age[t, k] is the trading-day distance from the t-th market day in
lookback window to the k-th event. This lets financial reports affect the
following 20/60/120 trading days without using ffill as a fake daily feature.

Typical usage after your notebook has built `final`:

    final.write_parquet("/data/xujiayi/xjy/research_factors/model_specific/ered/final_events.parquet")

Then run:

    python build_ered_event_dataset.py \
        --final-path /data/xujiayi/xjy/research_factors/model_specific/ered/final_events.parquet \
        --dates-path /data/xujiayi/end2end/axis/dates.npy \
        --ticks-path /data/xujiayi/end2end/axis/ticks.npy \
        --out-dir /data/xujiayi/xjy/research_factors/model_specific/ered/event_store
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_RAW_FEATURES: List[str] = [
    "basic_eps_yoy",
    "total_operating_revenue_yoy",
    "np_parent_company_owners_yoy",
    "net_operate_cash_flow_yoy",
    "roe_yoy",
    "l2a_yoy",
    "gpm_yoy",
    "npm_yoy",
    "basic_eps_qoq",
    "total_operating_revenue_qoq",
    "np_parent_company_owners_qoq",
    "net_operate_cash_flow_qoq",
    "roe_qoq",
    "l2a_qoq",
    "gpm_qoq",
    "npm_qoq",
    "eps_ue_sue",
    "or_ue_sue",
    "np_ue_sue",
    "roe_ue_sue",
    "log_distance",
]

DEFAULT_EVENT_TYPE_COLS: List[str] = ["mask1", "mask2", "mask3", "mask4"]
DEFAULT_EVENT_TYPE_NAMES: List[str] = ["q1", "q2", "q3", "q4"]


@dataclass
class EventStoreConfig:
    max_events: int = 4
    cross_sectional_standardize: bool = True
    add_missing_flags: bool = True
    add_static_time_features: bool = True
    dtype: str = "float32"


def _ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_table(path: str | os.PathLike[str]) -> pd.DataFrame:
    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"unsupported table format: {path}")


def _normalize_dates(dates: Sequence) -> pd.DatetimeIndex:
    return pd.to_datetime(pd.Series(dates)).dt.normalize()


def _to_datetime_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _standardize_by_group(df: pd.DataFrame, cols: Sequence[str], group_col: str) -> pd.DataFrame:
    """Cross-sectionally standardize event features by effective trading day."""
    out = df.copy()
    for col in cols:
        x = pd.to_numeric(out[col], errors="coerce")
        mean = x.groupby(out[group_col]).transform("mean")
        std = x.groupby(out[group_col]).transform(lambda v: float(np.nanstd(v.to_numpy(dtype=float), ddof=0)))
        std = std.replace(0, np.nan)
        out[col] = (x - mean) / std
    return out


def _infer_event_type_matrix(final: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Use mask1-mask4 if present; otherwise infer quarter from report_date month."""
    if all(c in final.columns for c in DEFAULT_EVENT_TYPE_COLS):
        arr = final[DEFAULT_EVENT_TYPE_COLS].fillna(0).astype(float).to_numpy()
        # Defensive fallback: if all masks are zero for a row, infer from report_date.
        empty = arr.sum(axis=1) == 0
        if empty.any() and "report_date" in final.columns:
            months = _to_datetime_series(final.loc[empty, "report_date"]).dt.month.to_numpy()
            inferred = np.zeros((empty.sum(), 4), dtype=float)
            month_to_idx = {3: 0, 6: 1, 9: 2, 12: 3}
            for i, m in enumerate(months):
                if int(m) in month_to_idx:
                    inferred[i, month_to_idx[int(m)]] = 1.0
            arr[empty] = inferred
        return arr, [f"event_type_{n}" for n in DEFAULT_EVENT_TYPE_NAMES]

    if "report_date" not in final.columns:
        raise ValueError("need either mask1-mask4 columns or report_date to infer event type")
    months = _to_datetime_series(final["report_date"]).dt.month.to_numpy()
    arr = np.zeros((len(final), 4), dtype=float)
    month_to_idx = {3: 0, 6: 1, 9: 2, 12: 3}
    for i, m in enumerate(months):
        if pd.notna(m) and int(m) in month_to_idx:
            arr[i, month_to_idx[int(m)]] = 1.0
    return arr, [f"event_type_{n}" for n in DEFAULT_EVENT_TYPE_NAMES]


def build_event_store(
    final: pd.DataFrame,
    dates: Sequence,
    ticks: Sequence,
    out_dir: str | os.PathLike[str],
    raw_feature_names: Optional[Sequence[str]] = None,
    cfg: EventStoreConfig = EventStoreConfig(),
) -> Dict[str, object]:
    """
    Build event store arrays from a `final` table produced by your financial-report pipeline.

    Required columns:
        tick, report_date, sheet_infodate

    Recommended feature columns are DEFAULT_RAW_FEATURES. Missing feature columns
    are created as NaN, then represented by value=0 plus missing flag.
    """
    raw_feature_names = list(raw_feature_names or DEFAULT_RAW_FEATURES)
    out_path = _ensure_dir(out_dir)

    final = final.copy()
    required = ["tick", "report_date", "sheet_infodate"]
    missing_required = [c for c in required if c not in final.columns]
    if missing_required:
        raise ValueError(f"final is missing required columns: {missing_required}")

    dates_idx = _normalize_dates(dates)
    dates_np = dates_idx.to_numpy(dtype="datetime64[ns]")
    ticks_arr = np.asarray(ticks).astype(str)
    tick_to_idx = {t: i for i, t in enumerate(ticks_arr)}

    final["tick"] = final["tick"].astype(str)
    final["report_date"] = _to_datetime_series(final["report_date"])
    final["sheet_infodate"] = _to_datetime_series(final["sheet_infodate"])
    final["tick_idx"] = final["tick"].map(tick_to_idx)

    # Announcement on non-trading day becomes visible on the next trading day.
    info_np = final["sheet_infodate"].to_numpy(dtype="datetime64[ns]")
    effective_idx = np.searchsorted(dates_np, info_np, side="left")
    final["effective_idx"] = effective_idx

    valid = (
        final["tick_idx"].notna()
        & final["report_date"].notna()
        & final["sheet_infodate"].notna()
        & (final["effective_idx"] >= 0)
        & (final["effective_idx"] < len(dates_idx))
    )
    final = final.loc[valid].copy()
    final["tick_idx"] = final["tick_idx"].astype(int)
    final["effective_idx"] = final["effective_idx"].astype(int)

    # If the same stock/report_date has multiple publication rows, keep the first
    # PIT-visible one. If there are distinct events on the same day, they remain
    # separate as long as report_date differs.
    final = final.sort_values(["tick_idx", "report_date", "sheet_infodate", "effective_idx"])
    final = final.drop_duplicates(subset=["tick_idx", "report_date"], keep="first")

    for col in raw_feature_names:
        if col not in final.columns:
            final[col] = np.nan
        final[col] = pd.to_numeric(final[col], errors="coerce")

    if cfg.cross_sectional_standardize:
        final = _standardize_by_group(final, raw_feature_names, "effective_idx")

    event_type_values, event_type_names = _infer_event_type_matrix(final)

    raw_values = final[raw_feature_names].to_numpy(dtype=float)
    missing_values = np.isnan(raw_values).astype(float)
    raw_values = np.nan_to_num(raw_values, nan=0.0, posinf=0.0, neginf=0.0)

    parts = [event_type_values, raw_values]
    feature_names = event_type_names + list(raw_feature_names)

    if cfg.add_missing_flags:
        parts.append(missing_values)
        feature_names += [f"{c}_is_missing" for c in raw_feature_names]

    if cfg.add_static_time_features:
        report_lag_days = (final["sheet_infodate"] - final["report_date"]).dt.days.clip(lower=0).fillna(0).to_numpy(dtype=float)
        report_month = final["report_date"].dt.month.fillna(0).to_numpy(dtype=float)
        static_time = np.column_stack([
            np.log1p(report_lag_days),
            report_month / 12.0,
        ])
        parts.append(static_time)
        feature_names += ["log1p_report_lag_days", "report_month_div_12"]

    event_x = np.concatenate(parts, axis=1).astype(cfg.dtype)

    event_tick_idx = final["tick_idx"].to_numpy(dtype=np.int32)
    event_effective_idx = final["effective_idx"].to_numpy(dtype=np.int32)
    event_report_ordinal = final["report_date"].map(lambda x: x.toordinal()).to_numpy(dtype=np.int32)

    # Sort by stock, then PIT effective date. Keep stable order for same-day multi-events.
    order = np.lexsort((np.arange(len(event_tick_idx)), event_effective_idx, event_tick_idx))
    event_x = event_x[order]
    event_tick_idx = event_tick_idx[order]
    event_effective_idx = event_effective_idx[order]
    event_report_ordinal = event_report_ordinal[order]
    final_sorted = final.iloc[order].reset_index(drop=True)

    n_ticks = len(ticks_arr)
    tick_ptr = np.zeros(n_ticks + 1, dtype=np.int64)
    counts = np.bincount(event_tick_idx, minlength=n_ticks)
    tick_ptr[1:] = np.cumsum(counts)

    np.save(out_path / "event_x.npy", event_x)
    np.save(out_path / "event_tick_idx.npy", event_tick_idx)
    np.save(out_path / "event_effective_idx.npy", event_effective_idx)
    np.save(out_path / "event_report_ordinal.npy", event_report_ordinal)
    np.save(out_path / "event_tick_ptr.npy", tick_ptr)
    np.save(out_path / "dates.npy", dates_idx.astype(str).to_numpy())
    np.save(out_path / "ticks.npy", ticks_arr)

    # Save a human-readable long table for debugging / checking PIT logic.
    final_sorted[["tick", "report_date", "sheet_infodate", "tick_idx", "effective_idx"] + [c for c in raw_feature_names if c in final_sorted.columns]].to_csv(
        out_path / "events_long_debug.csv", index=False
    )

    metadata = {
        "config": asdict(cfg),
        "event_dim": int(event_x.shape[1]),
        "num_events": int(event_x.shape[0]),
        "num_ticks": int(n_ticks),
        "feature_names": feature_names,
        "raw_feature_names": list(raw_feature_names),
        "event_type_names": event_type_names,
        "note": "event_x is PIT long-format. Dataset retrieves last K events by effective_idx <= trade_idx and builds event_age [L,K].",
    }
    with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata


class PITEventStore:
    """Fast retrieval of last K PIT-visible events for a given stock/date sample."""

    def __init__(self, store_dir: str | os.PathLike[str], max_events: Optional[int] = None):
        self.store_dir = Path(store_dir)
        with open(self.store_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.event_x = np.load(self.store_dir / "event_x.npy", mmap_mode="r")
        self.event_tick_idx = np.load(self.store_dir / "event_tick_idx.npy", mmap_mode="r")
        self.event_effective_idx = np.load(self.store_dir / "event_effective_idx.npy", mmap_mode="r")
        self.event_report_ordinal = np.load(self.store_dir / "event_report_ordinal.npy", mmap_mode="r")
        self.tick_ptr = np.load(self.store_dir / "event_tick_ptr.npy", mmap_mode="r")
        self.max_events = int(max_events or self.metadata["config"].get("max_events", 4))
        self.event_dim = int(self.metadata["event_dim"])

    def get_event_window(
        self,
        tick_idx: int,
        trade_idx: int,
        lookback: int,
        max_events: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        K = int(max_events or self.max_events)
        event_x = np.zeros((K, self.event_dim), dtype=np.float32)
        event_mask = np.zeros((K,), dtype=np.int64)
        event_age = np.full((lookback, K), -1, dtype=np.int64)

        lo = int(self.tick_ptr[tick_idx])
        hi = int(self.tick_ptr[tick_idx + 1])
        if hi <= lo:
            return event_x, event_mask, event_age

        eff = self.event_effective_idx[lo:hi]
        # Because the store is sorted by effective_idx within each tick.
        pos = int(np.searchsorted(eff, trade_idx, side="right"))
        if pos <= 0:
            return event_x, event_mask, event_age

        selected = np.arange(lo, lo + pos, dtype=np.int64)[-K:]
        m = len(selected)
        event_x[:m] = np.asarray(self.event_x[selected], dtype=np.float32)
        event_mask[:m] = 1

        window_idx = np.arange(trade_idx - lookback + 1, trade_idx + 1, dtype=np.int64)
        selected_eff = np.asarray(self.event_effective_idx[selected], dtype=np.int64)
        event_age[:, :m] = window_idx[:, None] - selected_eff[None, :]
        return event_x, event_mask, event_age


def load_memmap_matrix(path: str | os.PathLike[str], shape: Tuple[int, int], dtype: str | np.dtype = "float64") -> np.memmap:
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def build_future_return_target(close: np.ndarray, horizon: int = 5, industry_ret: Optional[np.ndarray] = None) -> np.ndarray:
    """Simple future return target. Optionally subtract industry/index return matrix."""
    close = np.asarray(close, dtype=float)
    target = np.full_like(close, np.nan, dtype=float)
    future = close[horizon:]
    now = close[:-horizon]
    ret = future / now - 1.0
    ret[~np.isfinite(ret)] = np.nan
    target[:-horizon] = ret
    if industry_ret is not None:
        target = target - industry_ret
    return target


def build_sample_index(
    target: np.ndarray,
    lookback: int,
    horizon: int = 0,
    valid_universe: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return [num_samples, 2] array: columns are trade_idx, tick_idx."""
    T, N = target.shape
    end = T - max(horizon, 0)
    rows: List[np.ndarray] = []
    for t in range(lookback - 1, end):
        valid = np.isfinite(target[t])
        if valid_universe is not None:
            valid = valid & valid_universe[t].astype(bool)
        idx = np.flatnonzero(valid)
        if idx.size:
            tt = np.full(idx.size, t, dtype=np.int32)
            rows.append(np.column_stack([tt, idx.astype(np.int32)]))
    if not rows:
        return np.empty((0, 2), dtype=np.int32)
    return np.vstack(rows).astype(np.int32)


class EREDPITWindowDataset(Dataset):
    """
    PyTorch Dataset aligned with PITEventResidualAlphaModel.

    market_arrays: dict of name -> [T,N] numpy/memmap matrices.
    target:        [T,N] matrix. target[trade_idx, tick_idx] is y.
    sample_index:  [S,2], each row = (trade_idx, tick_idx).
    """

    def __init__(
        self,
        market_arrays: Dict[str, np.ndarray],
        target: np.ndarray,
        event_store: PITEventStore,
        sample_index: np.ndarray,
        lookback: int = 60,
        max_events: Optional[int] = None,
        fill_value: float = 0.0,
    ):
        if not market_arrays:
            raise ValueError("market_arrays cannot be empty")
        self.market_names = list(market_arrays.keys())
        self.market_arrays = [market_arrays[k] for k in self.market_names]
        self.target = target
        self.event_store = event_store
        self.sample_index = np.asarray(sample_index, dtype=np.int32)
        self.lookback = int(lookback)
        self.max_events = max_events or event_store.max_events
        self.fill_value = float(fill_value)

    def __len__(self) -> int:
        return int(self.sample_index.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        trade_idx, tick_idx = self.sample_index[i]
        trade_idx = int(trade_idx)
        tick_idx = int(tick_idx)
        start = trade_idx - self.lookback + 1
        end = trade_idx + 1
        if start < 0:
            raise IndexError("trade_idx is too early for the configured lookback")

        cols = []
        for arr in self.market_arrays:
            x = np.asarray(arr[start:end, tick_idx], dtype=np.float32)
            cols.append(x)
        market_x = np.stack(cols, axis=-1)
        market_x = np.nan_to_num(market_x, nan=self.fill_value, posinf=self.fill_value, neginf=self.fill_value)
        market_mask = np.ones((self.lookback,), dtype=np.int64)

        event_x, event_mask, event_age = self.event_store.get_event_window(
            tick_idx=tick_idx,
            trade_idx=trade_idx,
            lookback=self.lookback,
            max_events=self.max_events,
        )
        y = np.asarray(self.target[trade_idx, tick_idx], dtype=np.float32)
        stock_mask = np.asarray(1 if np.isfinite(y) else 0, dtype=np.int64)
        if not np.isfinite(y):
            y = np.asarray(0.0, dtype=np.float32)

        return {
            "market_x": torch.from_numpy(market_x).float(),
            "market_mask": torch.from_numpy(market_mask).long(),
            "event_x": torch.from_numpy(event_x).float(),
            "event_mask": torch.from_numpy(event_mask).long(),
            "event_age": torch.from_numpy(event_age).long(),
            "y": torch.tensor(y).float(),
            "stock_mask": torch.tensor(stock_mask).long(),
            "trade_idx": torch.tensor(trade_idx).long(),
            "tick_idx": torch.tensor(tick_idx).long(),
        }


def save_sample_index(path: str | os.PathLike[str], sample_index: np.ndarray) -> None:
    np.save(path, np.asarray(sample_index, dtype=np.int32))


def load_sample_index(path: str | os.PathLike[str]) -> np.ndarray:
    return np.load(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PIT ERED event store from final financial-report table.")
    parser.add_argument("--final-path", required=True, help="Parquet/CSV/Pickle table with tick, report_date, sheet_infodate and features.")
    parser.add_argument("--dates-path", required=True, help="axis dates.npy")
    parser.add_argument("--ticks-path", required=True, help="axis ticks.npy")
    parser.add_argument("--out-dir", required=True, help="Output event store directory")
    parser.add_argument("--max-events", type=int, default=4)
    parser.add_argument("--no-standardize", action="store_true", help="Disable cross-sectional standardization by effective date.")
    args = parser.parse_args()

    final = _load_table(args.final_path)
    dates = np.load(args.dates_path, allow_pickle=True)
    ticks = np.load(args.ticks_path, allow_pickle=True)
    cfg = EventStoreConfig(
        max_events=args.max_events,
        cross_sectional_standardize=not args.no_standardize,
    )
    meta = build_event_store(final=final, dates=dates, ticks=ticks, out_dir=args.out_dir, cfg=cfg)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
