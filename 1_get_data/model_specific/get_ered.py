# -*- coding: utf-8 -*-
"""
生成 ERED 事件流输入。

当前数据语义：
- 财报事件长表只包含真实存在的财报事件，不补空季度；
- event_x = 长表中的财报特征值；
- event_tick = 长表中的 tick_idx；
- event_effective_idx = 长表中的 date_idx，也就是报告发布日映射到交易日后的 index；
- event_mask 不在这里生成，由 dataset.eventstore 在取最近 K 个事件时动态生成。
"""

import os
import sys
from pathlib import Path

project_root = os.path.abspath("..")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import polars as pl
from sqlalchemy import create_engine
import pymssql


# =========================
# Config
# =========================
start_dt = "2008-01-01"
end_dt = "2025-12-31"
ROOT = Path("/data/xujiayi/end2end/")
OUT = Path("/data/xujiayi/xjy/research_factors/model_input/ered_v2/")
OUT.mkdir(parents=True, exist_ok=True)

JY_CONFIG = {
    "server": "10.10.0.102",
    "user": "jydbReader",
    "password": "jy@9043!Reader",
    "database": "jydb",
    "charset": "cp936",
}

# 如后续需要分钟/高频库，可继续使用该连接；当前事件流生成不依赖它。
STR_CONN = create_engine(
    "mysql+pymysql://QuantReader:Quant%40Reader%21zsfund.com@10.10.6.101:9030/HighFrequency"
)


# =========================
# Axis / masks
# =========================
dates = np.load(ROOT / "axis" / "dates.npy", allow_pickle=True)
ticks = np.load(ROOT / "axis" / "ticks.npy", allow_pickle=True)
dates = pd.to_datetime(dates).normalize().to_numpy()

close = np.memmap(
    ROOT / "d_field" / "close.bin",
    dtype=float,
    shape=(len(dates), len(ticks)),
    mode="r",
)
nan_mask = np.isnan(close)

industry = np.memmap(
    "/data/xujiayi/xjy/mask/industry.bin",
    shape=(len(dates), len(ticks)),
    mode="r",
    dtype=float,
)


# =========================
# Load fundamental event table
# =========================
sql_f = f'''
select
    C.SecuCode as "tick",
    A.EndDate as "report_date",
    A.InfoPublDate as "date",
    A.ROETTM,
    A.ROICTTM,
    A.GrossIncomeRatioTTM,
    A.NetProfitRatioTTM,
    A.PeriodCostsRateTTM,
    A.AdminiExpenseRateTTM,
    A.TotalAssetTRateTTM,
    A.ARTRate,
    A.InventoryTRate,
    A.DebtAssetsRatio,
    A.LongDebtRatio,
    A.NPParentCompanyCutYOY,
    A.TotalAssetGrowRate,
    A.NetOperateCashFlowYOY,
    A.NOCFToOperatingNITTM,
    A.SaleServiceCashToORTTM,
    A.OperCashInToAsset,
    A.FixAssetRatio,
    A.IntangibleAssetRatio,
    A.DividendPaidRatio,
    A.RetainedEarningRatio
from LC_MainIndexNew A
left join SecuMain C
on A.CompanyCode = C.CompanyCode
where A.InfoPublDate <= '{end_dt}'
    and C.SecuMarket in (83,90)
    and C.SecuCategory=1

union all

select
    C.SecuCode as "tick",
    B.EndDate as "report_date",
    B.InfoPublDate as "date",
    B.ROETTM,
    B.ROICTTM,
    B.GrossIncomeRatioTTM,
    B.NetProfitRatioTTM,
    B.PeriodCostsRateTTM,
    B.AdminiExpenseRateTTM,
    B.TotalAssetTRateTTM,
    B.ARTRate,
    B.InventoryTRate,
    B.DebtAssetsRatio,
    B.LongDebtRatio,
    B.NPParentCompanyCutYOY,
    B.TotalAssetGrowRate,
    B.NetOperateCashFlowYOY,
    B.NOCFToOperatingNITTM,
    B.SaleServiceCashToORTTM,
    B.OperCashInToAsset,
    B.FixAssetRatio,
    B.IntangibleAssetRatio,
    B.DividendPaidRatio,
    B.RetainedEarningRatio
from LC_STIBMainIndex B
left join SecuMain C
on B.CompanyCode = C.CompanyCode
where B.InfoPublDate <= '{end_dt}'
    and C.SecuMarket in (83,90)
    and C.SecuCategory=1
    and B.IfMerged=1
    and B.IfAdjusted=2
'''

with pymssql.connect(**JY_CONFIG) as jy_conn:
    f = pd.read_sql(sql_f, jy_conn)

f = pl.from_pandas(f)
f = (
    f.sort(["tick", "report_date", "date"])
    .filter(pl.col("tick").is_in(ticks))
    .filter(pl.col("report_date") >= pl.datetime(2007, 12, 31))
    .unique(subset=["tick", "date"], keep="last")
    .unique(subset=["tick", "report_date"], keep="first")
)

feat_cols = [
    "ROETTM",
    "ROICTTM",
    "GrossIncomeRatioTTM",
    "NetProfitRatioTTM",
    "PeriodCostsRateTTM",
    "AdminiExpenseRateTTM",
    "TotalAssetTRateTTM",
    "ARTRate",
    "InventoryTRate",
    "DebtAssetsRatio",
    "LongDebtRatio",
    "NPParentCompanyCutYOY",
    "TotalAssetGrowRate",
    "NetOperateCashFlowYOY",
    "NOCFToOperatingNITTM",
    "SaleServiceCashToORTTM",
    "OperCashInToAsset",
    "FixAssetRatio",
    "IntangibleAssetRatio",
    # "DividendPaidRatio",
    # "RetainedEarningRatio",
]

f = f.select(["tick", "report_date", "date"] + feat_cols)


# =========================
# Map publish date to trade date / index
# =========================
calendar = pl.DataFrame({"trade_date": dates})
df = (
    f.sort("date")
    .join_asof(calendar, left_on="date", right_on="trade_date", strategy="forward")
    .sort(["tick", "date"])
)
df = df.sort(["tick", "trade_date", "date"]).unique(
    subset=["tick", "trade_date"], keep="last"
)

# date_idx 就是事件 effective_idx；tick_idx 就是 event_tick。
date2idx = {d: i for i, d in enumerate(dates)}
tick2idx = {t: i for i, t in enumerate(ticks)}

date_idx = np.array([date2idx.get(x, -1) for x in df["trade_date"].to_list()], dtype=np.int64)
tick_idx = np.array([tick2idx.get(x, -1) for x in df["tick"].to_list()], dtype=np.int64)

df = df.with_columns([
    pl.Series("date_idx", date_idx),
    pl.Series("tick_idx", tick_idx),
]).filter(
    (pl.col("date_idx") >= 0) & (pl.col("tick_idx") >= 0)
)


# =========================
# Feature cleaning: industry fill + winsorize + cross-sectional standardize
# =========================
ind = industry[df["date_idx"].to_numpy(), df["tick_idx"].to_numpy()]
df = df.with_columns(pl.Series("industry", ind))

for feat in feat_cols:
    # 行业中位数填补缺失
    industry_med = pl.col(feat).median().over(["trade_date", "industry"])
    df = df.with_columns(
        pl.when(pl.col(feat).is_null() & pl.col("industry").is_not_null())
        .then(industry_med)
        .otherwise(pl.col(feat))
        .alias(feat)
    )

    # 当日横截面 MAD 去极值
    market_med = pl.col(feat).median().over("trade_date")
    mad = (pl.col(feat) - market_med).abs().median().over("trade_date")
    upper = market_med + 3 * 1.4826 * mad
    lower = market_med - 3 * 1.4826 * mad
    df = df.with_columns(
        pl.when(pl.col(feat) > upper)
        .then(upper)
        .when(pl.col(feat) < lower)
        .then(lower)
        .otherwise(pl.col(feat))
        .alias(feat)
    )

    # 当日横截面标准化
    mean = pl.col(feat).mean().over("trade_date")
    std = pl.col(feat).std().over("trade_date")
    df = df.with_columns(
        pl.when((std == 0) | std.is_null())
        .then(0.0)
        .otherwise((pl.col(feat) - mean) / std)
        .alias(feat)
    )


# =========================
# Save event arrays for dataset.eventstore
# =========================
event_df = df.select(["tick_idx", "date_idx"] + feat_cols).sort(["tick_idx", "date_idx"])

event_x = event_df.select(feat_cols).to_numpy().astype(float).tofile(OUT/"event_x.bin")
event_tick = event_df["tick_idx"].to_numpy().astype(np.int64).tofile(OUT/"event_tick.bin")
event_effective_idx = event_df["date_idx"].to_numpy().astype(np.int64).tofile(OUT/"event_effective_idx.bin")

print(f"saved to: {OUT}")
print(f"event_x: {event_x.shape}")
print(f"event_tick: {event_tick.shape}")
print(f"event_effective_idx: {event_effective_idx.shape}")
print("event_x = 财报长表特征值")
print("event_tick = tick_idx")
print("event_effective_idx = date_idx")
