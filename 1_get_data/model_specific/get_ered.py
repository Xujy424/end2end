# -*- coding: utf-8 -*-
"""
生成 ERED 事件流数据。

目标：把“股票-报告日-报告发布日-财报特征”的长表，处理成 dataset.eventstore 可读取的
point-in-time 事件数组。

输出文件仅包含：
    event_x.npy               float32 [E, D_event]
    event_tick.npy            int64   [E]
    event_effective_idx.npy   int64   [E]

重要语义：
- 原始 event table 只包含真实存在的财报事件，不补空季度；
- event_mask 不在这里生成，而是在 dataset.eventstore 按“最近 K 个真实事件，不足 K padding”生成；
- event_effective_idx 是事件对模型可见的第一个交易日 index。若公告盘后发布，建议使用下一交易日。

用法示例：
python 1_get_data/model_specific/get_ered.py \
  --input /path/fundamental_events.parquet \
  --output /data/xujiayi/xjy/research_factors/model_specific/ERED \
  --axis-dir /data/xujiayi/end2end/axis \
  --stock-col tick \
  --report-date-col report_date \
  --publish-date-col date \
  --feature-cols ROETTM,ROICTTM,GrossIncomeRatioTTM,NetProfitRatioTTM
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    if suffix in [".feather", ".ft"]:
        return pd.read_feather(path)
    if suffix in [".csv", ".txt"]:
        return pd.read_csv(path)
    if suffix in [".pkl", ".pickle"]:
        return pd.read_pickle(path)
    raise ValueError(f"不支持的输入格式: {path}")


def load_axis(axis_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    axis_dir = Path(axis_dir)
    dates = pd.to_datetime(np.load(axis_dir / "dates.npy", allow_pickle=True)).normalize()
    ticks = np.load(axis_dir / "ticks.npy", allow_pickle=True)
    return np.asarray(dates), np.asarray(ticks)


def normalize_stock_code(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def infer_feature_cols(df: pd.DataFrame, exclude_cols: Iterable[str]) -> List[str]:
    exclude = set(exclude_cols)
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def map_effective_idx(publish_dates: pd.Series, trade_dates: np.ndarray, next_trade_day: bool = True) -> np.ndarray:
    publish_dates = pd.to_datetime(publish_dates, errors="coerce").dt.normalize().to_numpy()
    side = "right" if next_trade_day else "left"
    idx = np.searchsorted(trade_dates, publish_dates, side=side).astype(np.int64)
    idx[idx >= len(trade_dates)] = -1
    return idx


def add_delta_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    stock_col: str,
    report_date_col: str,
    yoy: bool = False,
    qoq: bool = False,
) -> tuple[pd.DataFrame, List[str]]:
    """可选生成事件序列上的同比/环比差分。

    注意：这里不补空季度。shift(4) 表示同股票已存在事件序列的前 4 条，
    如果你要求严格自然季度同比，建议在上游长表里先算好后作为 feature_cols 传入。
    """
    if not yoy and not qoq:
        return df, feature_cols

    df = df.sort_values([stock_col, report_date_col]).copy()
    group = df.groupby(stock_col, sort=False)
    new_cols: List[str] = []
    if yoy:
        for col in feature_cols:
            name = f"{col}_yoy_delta"
            df[name] = df[col] - group[col].shift(4)
            new_cols.append(name)
    if qoq:
        for col in feature_cols:
            name = f"{col}_qoq_delta"
            df[name] = df[col] - group[col].shift(1)
            new_cols.append(name)
    return df, feature_cols + new_cols


def robust_standardize(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """对事件特征做轻量稳健标准化。"""
    df = df.copy()
    for col in feature_cols:
        x = pd.to_numeric(df[col], errors="coerce").astype(float)
        lo = x.quantile(0.005)
        hi = x.quantile(0.995)
        x = x.clip(lo, hi)
        med = x.median()
        std = x.std(ddof=0)
        if not np.isfinite(std) or std < 1e-12:
            df[col] = 0.0
        else:
            df[col] = (x.fillna(med) - med) / std
    return df


def build_event_arrays(
    event_table: pd.DataFrame,
    axis_dir: str | Path,
    stock_col: str,
    report_date_col: str,
    publish_date_col: str,
    feature_cols: Optional[List[str]] = None,
    next_trade_day: bool = True,
    standardize: bool = True,
    derive_yoy: bool = False,
    derive_qoq: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trade_dates, ticks = load_axis(axis_dir)
    tick_to_idx = {normalize_stock_code(t): i for i, t in enumerate(ticks)}

    df = event_table.copy()
    df[stock_col] = df[stock_col].map(normalize_stock_code)
    df[report_date_col] = pd.to_datetime(df[report_date_col], errors="coerce")
    df[publish_date_col] = pd.to_datetime(df[publish_date_col], errors="coerce")
    df["event_tick"] = df[stock_col].map(tick_to_idx)
    df["event_effective_idx"] = map_effective_idx(df[publish_date_col], trade_dates, next_trade_day)

    df = df.dropna(subset=[stock_col, report_date_col, publish_date_col, "event_tick"])
    df["event_tick"] = df["event_tick"].astype(np.int64)
    df = df[df["event_effective_idx"] >= 0]

    if feature_cols is None:
        feature_cols = infer_feature_cols(
            df,
            exclude_cols=[stock_col, report_date_col, publish_date_col, "event_tick", "event_effective_idx"],
        )
    if not feature_cols:
        raise ValueError("未找到财报特征列，请用 --feature-cols 指定")

    for col in feature_cols:
        if col not in df.columns:
            raise KeyError(f"feature column not found: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df, feature_cols = add_delta_features(df, feature_cols, stock_col, report_date_col, derive_yoy, derive_qoq)
    if standardize:
        df = robust_standardize(df, feature_cols)
    else:
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 同股票、同报告期、同发布生效日重复时，保留最后一条。仍然只保留真实事件。
    df = df.sort_values(["event_tick", "event_effective_idx", report_date_col, publish_date_col])
    df = df.drop_duplicates(["event_tick", "event_effective_idx", report_date_col], keep="last")

    report_key = pd.to_datetime(df[report_date_col]).view("int64").to_numpy()
    event_tick = df["event_tick"].to_numpy(dtype=np.int64)
    event_effective_idx = df["event_effective_idx"].to_numpy(dtype=np.int64)
    event_x = df[feature_cols].to_numpy(dtype=np.float32)

    # 保证每只股票内部按生效日升序，dataset 中 searchsorted 才可靠。
    order = np.lexsort((report_key, event_effective_idx, event_tick))
    return event_x[order], event_tick[order], event_effective_idx[order]


def save_event_arrays(output_dir: str | Path, event_x: np.ndarray, event_tick: np.ndarray, event_effective_idx: np.ndarray) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "event_x.npy", event_x.astype(np.float32, copy=False))
    np.save(output_dir / "event_tick.npy", event_tick.astype(np.int64, copy=False))
    np.save(output_dir / "event_effective_idx.npy", event_effective_idx.astype(np.int64, copy=False))
    print(f"saved to: {output_dir}")
    print(f"event_x: {event_x.shape}")
    print(f"event_tick: {event_tick.shape}")
    print(f"event_effective_idx: {event_effective_idx.shape}")


def build_from_file(
    input_path: str | Path,
    output_dir: str | Path,
    axis_dir: str | Path,
    stock_col: str,
    report_date_col: str,
    publish_date_col: str,
    feature_cols: Optional[List[str]] = None,
    next_trade_day: bool = True,
    standardize: bool = True,
    derive_yoy: bool = False,
    derive_qoq: bool = False,
) -> None:
    table = read_table(input_path)
    event_x, event_tick, event_effective_idx = build_event_arrays(
        table,
        axis_dir=axis_dir,
        stock_col=stock_col,
        report_date_col=report_date_col,
        publish_date_col=publish_date_col,
        feature_cols=feature_cols,
        next_trade_day=next_trade_day,
        standardize=standardize,
        derive_yoy=derive_yoy,
        derive_qoq=derive_qoq,
    )
    save_event_arrays(output_dir, event_x, event_tick, event_effective_idx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ERED event arrays from a fundamental event table")
    parser.add_argument("--input", required=True, help="事件长表路径：csv/parquet/feather/pkl")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--axis-dir", required=True, help="包含 dates.npy/ticks.npy 的目录")
    parser.add_argument("--stock-col", required=True)
    parser.add_argument("--report-date-col", required=True)
    parser.add_argument("--publish-date-col", required=True)
    parser.add_argument("--feature-cols", default=None, help="逗号分隔；不传则自动取数值列")
    parser.add_argument("--same-day-effective", action="store_true", help="发布日当天可见；默认下一交易日可见")
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--derive-yoy", action="store_true")
    parser.add_argument("--derive-qoq", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    feature_cols = None if args.feature_cols is None else [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    build_from_file(
        input_path=args.input,
        output_dir=args.output,
        axis_dir=args.axis_dir,
        stock_col=args.stock_col,
        report_date_col=args.report_date_col,
        publish_date_col=args.publish_date_col,
        feature_cols=feature_cols,
        next_trade_day=not args.same_day_effective,
        standardize=not args.no_standardize,
        derive_yoy=args.derive_yoy,
        derive_qoq=args.derive_qoq,
    )
