# -*- coding: utf-8 -*-
"""
构建 ERED 财报事件数据。

输入：股票-报告日-报告发布日-财报特征的长表。
输出：dataset.eventstore.PITEventStore 可直接读取的 point-in-time 事件数组。

输出文件：
    metadata.json
    feature_cols.json
    event_x.npy               float32, [E, D_event]
    event_tick.npy            int64,   [E]
    event_effective_idx.npy   int64,   [E]
    event_report_idx.npy      int64,   [E]
    event_tick_ptr.npy        int64,   [S+1]，兼容旧 CSR 读取方式/排查

核心原则：
    1. report_period / 报告日不是模型可见日，只用于排序、排重和派生同比；
    2. publish_date / 报告发布日期需要映射成 effective_idx；
    3. 如果公告通常盘后发布，effective_idx 应取下一个交易日；
    4. 每只股票内部按 effective_idx 升序排序，保证 searchsorted 语义正确。
"""

from __future__ import annotations

import argparse
import json
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


def map_publish_to_effective_idx(
    publish_dates: pd.Series,
    trade_dates: np.ndarray,
    use_next_trade_day: bool = True,
) -> np.ndarray:
    """把报告发布日期映射到模型可见的交易日 index。"""

    publish_dates = pd.to_datetime(publish_dates, errors="coerce").dt.normalize().to_numpy()
    side = "right" if use_next_trade_day else "left"
    idx = np.searchsorted(trade_dates, publish_dates, side=side).astype(np.int64)
    idx[idx >= len(trade_dates)] = -1
    return idx


def map_report_period_idx(report_dates: pd.Series, trade_dates: np.ndarray) -> np.ndarray:
    """把报告期映射到不晚于该报告日的最近交易日，仅用于追踪。"""

    report_dates = pd.to_datetime(report_dates, errors="coerce").dt.normalize().to_numpy()
    idx = np.searchsorted(trade_dates, report_dates, side="right").astype(np.int64) - 1
    idx[(idx < 0) | (idx >= len(trade_dates))] = -1
    return idx


def infer_feature_cols(df: pd.DataFrame, exclude_cols: Iterable[str]) -> List[str]:
    exclude = set(exclude_cols)
    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    return feature_cols


def add_derived_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    stock_col: str,
    report_period_col: str,
    derive_yoy: bool = False,
    derive_qoq: bool = False,
) -> tuple[pd.DataFrame, List[str]]:
    """可选生成简单同比/环比差分特征。"""

    df = df.sort_values([stock_col, report_period_col]).copy()
    new_cols: List[str] = []
    grouped = df.groupby(stock_col, sort=False)

    if derive_yoy:
        for col in feature_cols:
            new_col = f"{col}_yoy_delta"
            df[new_col] = df[col] - grouped[col].shift(4)
            new_cols.append(new_col)

    if derive_qoq:
        for col in feature_cols:
            new_col = f"{col}_qoq_delta"
            df[new_col] = df[col] - grouped[col].shift(1)
            new_cols.append(new_col)

    return df, feature_cols + new_cols


def winsorize_standardize(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """全样本去极值 + robust 标准化。

    如果要完全避免 scaler 使用未来信息，应在外部按训练集拟合分位数和标准差，
    再应用到验证/测试。这里作为 notebook 脚本的默认版本，先保证稳定可用。
    """

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


def build_event_tick_ptr(event_tick: np.ndarray, num_ticks: int) -> np.ndarray:
    counts = np.bincount(event_tick, minlength=num_ticks).astype(np.int64)
    ptr = np.zeros(num_ticks + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(counts)
    return ptr


def build_ered_event_data(
    input_path: str | Path,
    output_dir: str | Path,
    axis_dir: str | Path,
    stock_col: str,
    report_period_col: str,
    publish_date_col: str,
    feature_cols: Optional[List[str]] = None,
    use_next_trade_day: bool = True,
    derive_yoy: bool = False,
    derive_qoq: bool = False,
    standardize: bool = True,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trade_dates, ticks = load_axis(axis_dir)
    tick_to_idx = {normalize_stock_code(t): i for i, t in enumerate(ticks)}

    df = read_table(input_path).copy()
    df[stock_col] = df[stock_col].map(normalize_stock_code)
    df[report_period_col] = pd.to_datetime(df[report_period_col], errors="coerce")
    df[publish_date_col] = pd.to_datetime(df[publish_date_col], errors="coerce")

    df["event_tick"] = df[stock_col].map(tick_to_idx)
    df["event_effective_idx"] = map_publish_to_effective_idx(
        df[publish_date_col], trade_dates, use_next_trade_day=use_next_trade_day
    )
    df["event_report_idx"] = map_report_period_idx(df[report_period_col], trade_dates)

    df = df.dropna(subset=["event_tick", report_period_col, publish_date_col])
    df["event_tick"] = df["event_tick"].astype(np.int64)
    df = df[df["event_effective_idx"] >= 0]

    if feature_cols is None:
        feature_cols = infer_feature_cols(
            df,
            exclude_cols=[
                stock_col,
                report_period_col,
                publish_date_col,
                "event_tick",
                "event_effective_idx",
                "event_report_idx",
            ],
        )
    if not feature_cols:
        raise ValueError("没有找到可用财报特征列，请显式传入 --feature-cols")

    for col in feature_cols:
        if col not in df.columns:
            raise KeyError(f"feature column not found: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if derive_yoy or derive_qoq:
        df, feature_cols = add_derived_features(
            df,
            feature_cols=feature_cols,
            stock_col=stock_col,
            report_period_col=report_period_col,
            derive_yoy=derive_yoy,
            derive_qoq=derive_qoq,
        )

    if standardize:
        df = winsorize_standardize(df, feature_cols)
    else:
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 同一股票、同一生效日、同一报告期的重复记录保留最后一条。
    df = df.sort_values(["event_tick", "event_effective_idx", report_period_col, publish_date_col])
    df = df.drop_duplicates(["event_tick", "event_effective_idx", report_period_col], keep="last")

    event_x = df[feature_cols].to_numpy(dtype=np.float32)
    event_tick = df["event_tick"].to_numpy(dtype=np.int64)
    event_effective_idx = df["event_effective_idx"].to_numpy(dtype=np.int64)
    event_report_idx = df["event_report_idx"].to_numpy(dtype=np.int64)

    # 排序后每只股票内部 effective_idx 单调，searchsorted 才可靠。
    order = np.lexsort((event_report_idx, event_effective_idx, event_tick))
    event_x = event_x[order]
    event_tick = event_tick[order]
    event_effective_idx = event_effective_idx[order]
    event_report_idx = event_report_idx[order]
    event_tick_ptr = build_event_tick_ptr(event_tick, num_ticks=len(ticks))

    np.save(output_dir / "event_x.npy", event_x)
    np.save(output_dir / "event_tick.npy", event_tick)
    np.save(output_dir / "event_effective_idx.npy", event_effective_idx)
    np.save(output_dir / "event_report_idx.npy", event_report_idx)
    np.save(output_dir / "event_tick_ptr.npy", event_tick_ptr)

    metadata = {
        "event_dim": int(event_x.shape[1]),
        "num_events": int(event_x.shape[0]),
        "num_ticks": int(len(ticks)),
        "stock_col": stock_col,
        "report_period_col": report_period_col,
        "publish_date_col": publish_date_col,
        "use_next_trade_day": bool(use_next_trade_day),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(output_dir / "feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    print(f"完成: {output_dir}")
    print(f"事件数: {event_x.shape[0]}")
    print(f"特征数: {event_x.shape[1]}")
    print(f"股票数: {len(ticks)}")
    print(f"effective 规则: {'发布日后下一交易日' if use_next_trade_day else '发布日当天或之后首个交易日'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 ERED 财报事件数据")
    parser.add_argument("--input", required=True, help="财报长表路径，支持 csv/parquet/feather/pkl")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--axis-dir", required=True, help="包含 dates.npy 和 ticks.npy 的 axis 目录")
    parser.add_argument("--stock-col", required=True, help="股票代码列名")
    parser.add_argument("--report-period-col", required=True, help="报告日/报告期列名")
    parser.add_argument("--publish-date-col", required=True, help="报告发布日期列名")
    parser.add_argument("--feature-cols", default=None, help="逗号分隔的财报特征列；不传则自动取数值列")
    parser.add_argument("--same-day-effective", action="store_true", help="使用发布日当天或之后首个交易日作为生效日；默认使用下一交易日")
    parser.add_argument("--derive-yoy", action="store_true", help="生成 shift(4) 同比差分特征")
    parser.add_argument("--derive-qoq", action="store_true", help="生成 shift(1) 环比差分特征")
    parser.add_argument("--no-standardize", action="store_true", help="不做去极值和标准化")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    feature_cols = None if args.feature_cols is None else [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    build_ered_event_data(
        input_path=args.input,
        output_dir=args.output,
        axis_dir=args.axis_dir,
        stock_col=args.stock_col,
        report_period_col=args.report_period_col,
        publish_date_col=args.publish_date_col,
        feature_cols=feature_cols,
        use_next_trade_day=not args.same_day_effective,
        derive_yoy=args.derive_yoy,
        derive_qoq=args.derive_qoq,
        standardize=not args.no_standardize,
    )
