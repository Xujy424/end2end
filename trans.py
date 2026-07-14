# -*- coding: utf-8 -*-
"""
将原始财报数据 (tick, report_date, features) 转换为 PITEventStore 所需的格式。

输出文件：
    event_x.npy              : [总事件数, D_event]  所有财报的特征矩阵
    event_effective_idx.npy  : [总事件数]           每条财报对应的全局交易日索引
    event_tick_ptr.npy       : [总股票数 + 1]       每只股票在以上两个数组中的切分指针
    metadata.json            : 包含 event_dim 等元信息

使用方式：
    1. 修改下方的配置参数（数据路径、特征列名）
    2. 运行脚本
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json


# =============================================================================
# 配置区（请按实际情况修改）
# =============================================================================

# 原始数据路径（你的 CSV 或 Parquet）
RAW_DATA_PATH = "/path/to/your/raw_earnings_data.csv"

# 全局交易日历（必须与 BaseDataset 使用的 dates.npy 完全一致）
# 如果已有 dates.npy，直接加载：
TRADE_DATES_PATH = "/data/xujiayi/end2end/axis/dates.npy"

# 输出目录（存放生成的 event_x.npy 等文件）
OUTPUT_DIR = "/data/xujiayi/xjy/research_factors/model_specific/ered/event_store"

# 原始数据中，哪几列是财报特征（请换成你实际的列名）
FEATURE_COLS = [
    "basic_eps",
    "total_operating_revenue",
    "np_parent_company_owners",
    "operating_cost",
    "net_profit",
    "gpm",
    # ... 根据你的截图，还有更多 decimal 列，全部列在这里
]

# 每只股票最多保留多少个最近的事件（K）
MAX_EVENTS = 4


# =============================================================================
# 核心转换函数
# =============================================================================
def build_event_store(
    df_raw: pd.DataFrame,
    trade_dates: np.ndarray,
    feature_cols: list[str],
    output_dir: str,
    max_events: int = 4,
):
    """
    核心转换逻辑。
    
    Args:
        df_raw: 必须包含列 ['tick', 'report_date'] + feature_cols
        trade_dates: 字符串数组，如 ['2020-01-02', '2020-01-03', ...]
        feature_cols: 特征列名列表
        output_dir: 输出文件夹
        max_events: 每只股票最多保留 K 个事件（取最新的）
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Step 1: 清理原始数据
    # ------------------------------------------------------------
    df = df_raw[['tick', 'report_date'] + feature_cols].copy()
    
    # 日期统一转为字符串格式，方便对齐
    df['report_date'] = pd.to_datetime(df['report_date']).dt.strftime('%Y-%m-%d')
    
    # 特征列转为 float32（如果有缺失值，先填 0，后面还会统一处理）
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(np.float32)
    
    # 剔除报告日期为空的行
    df = df.dropna(subset=['report_date'])
    
    print(f"原始数据总行数（财报条数）: {len(df)}")

    # ------------------------------------------------------------
    # Step 2: 构建 报告日期 -> 交易日索引 的映射（关键：前向填充）
    # ------------------------------------------------------------
    # 很多财报的 report_date 是周末或节假日，不在交易日中。
    # 使用 pd.merge_asof 的 direction='forward'，将报告日期对齐到该日期之后的第一个交易日。
    # 含义：假设财报在 1月1日（假期）收盘后发布，那么它最早在 1月2日（交易日）生效。
    trade_df = pd.DataFrame({
        'trade_date': trade_dates,          # 交易日字符串
        'trade_idx': np.arange(len(trade_dates), dtype=np.int64)
    })
    
    # 必须先排序，merge_asof 要求 left 和 right 都按 key 排序
    df = df.sort_values('report_date')
    trade_df = trade_df.sort_values('trade_date')
    
    merged = pd.merge_asof(
        df,
        trade_df,
        left_on='report_date',
        right_on='trade_date',
        direction='forward',   # 向前找（即找未来的第一个交易日）
        allow_exact_matches=True,
    )
    
    # 如果某些财报的日期晚于最后一个交易日，会匹配不到，直接丢弃（通常数据不会这样）
    merged = merged.dropna(subset=['trade_idx'])
    merged['trade_idx'] = merged['trade_idx'].astype(np.int64)
    
    # 统计对齐情况
    print(f"成功对齐到交易日的财报数: {len(merged)}")
    if len(merged) < len(df):
        print(f"警告：有 {len(df) - len(merged)} 条财报无法对齐到交易日，已丢弃")

    # ------------------------------------------------------------
    # Step 3: 按 tick 分组，生成三个核心数组
    # ------------------------------------------------------------
    # 获取所有股票的 tick（按升序，保证 tick_ptr 与 BaseDataset.ticks 顺序一致）
    all_ticks = sorted(df_raw['tick'].unique())
    print(f"总股票数: {len(all_ticks)}")

    # 存放所有事件的特征和生效索引
    all_x = []           # 每个元素是一个 [N_i, D] 的矩阵
    all_eff_idx = []     # 每个元素是一个 [N_i] 的向量
    
    # tick_ptr 记录每只股票的起始位置，从 0 开始累加
    tick_ptr = [0]

    for tick in all_ticks:
        # 取出该股票的所有财报（已经对齐过 trade_idx）
        group = merged[merged['tick'] == tick]
        
        if len(group) == 0:
            # 该股票没有任何财报，指针保持不变（即占用 0 个事件）
            tick_ptr.append(tick_ptr[-1])
            continue
        
        # 按生效日（trade_idx）升序排序，确保是最早到最晚
        group = group.sort_values('trade_idx')
        
        # 取出生效索引和特征矩阵
        eff_idx = group['trade_idx'].values.astype(np.int64)
        features = group[feature_cols].values.astype(np.float32)
        
        # 注意：我们要取的是“最新的 K 个”，但这里为了保持时间顺序，先全部存入。
        # PITEventStore 在读取时会通过 [-K:] 自动截取最新的 K 个。
        # 因此这里存入全部，后续读取时由 store 控制 K。
        all_x.append(features)
        all_eff_idx.append(eff_idx)
        
        # 更新指针：累加该股票的事件数量
        tick_ptr.append(tick_ptr[-1] + len(features))

    # ------------------------------------------------------------
    # Step 4: 堆叠所有数据
    # ------------------------------------------------------------
    if len(all_x) == 0:
        # 极端情况：没有任何财报
        final_x = np.zeros((0, len(feature_cols)), dtype=np.float32)
        final_eff_idx = np.zeros((0,), dtype=np.int64)
    else:
        final_x = np.vstack(all_x)
        final_eff_idx = np.hstack(all_eff_idx)
    
    # 将 tick_ptr 转为 numpy 数组，int64 类型
    tick_ptr = np.array(tick_ptr, dtype=np.int64)

    print(f"总事件数: {len(final_eff_idx)}")
    print(f"事件特征维度: {final_x.shape[1]}")
    print(f"tick_ptr 长度（股票数+1）: {len(tick_ptr)}")

    # ------------------------------------------------------------
    # Step 5: 保存文件
    # ------------------------------------------------------------
    np.save(out_path / 'event_x.npy', final_x)
    np.save(out_path / 'event_effective_idx.npy', final_eff_idx)
    np.save(out_path / 'event_tick_ptr.npy', tick_ptr)
    
    # 保存元数据
    metadata = {
        "event_dim": len(feature_cols),
        "num_events": int(len(final_eff_idx)),
        "num_ticks": int(len(all_ticks)),
        "max_events_config": max_events,
        "config": {
            "max_events": max_events,
        }
    }
    with open(out_path / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 转换完成！文件已保存至: {out_path}")
    print("   - event_x.npy")
    print("   - event_effective_idx.npy")
    print("   - event_tick_ptr.npy")
    print("   - metadata.json")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    
    # 1. 加载原始数据
    print("正在加载原始数据...")
    df_raw = pd.read_csv(RAW_DATA_PATH)   # 如果是 parquet，用 pd.read_parquet
    
    # 2. 加载全局交易日历
    print("正在加载交易日历...")
    trade_dates = np.load(TRADE_DATES_PATH, allow_pickle=True)
    # 如果是字节串或 datetime64，统一转为字符串
    if trade_dates.dtype == np.dtype('<M8[us]') or trade_dates.dtype == np.dtype('datetime64'):
        trade_dates = pd.to_datetime(trade_dates).strftime('%Y-%m-%d')
    elif trade_dates.dtype == object:
        trade_dates = np.array([str(d) for d in trade_dates])
    
    print(f"交易日数量: {len(trade_dates)}")
    print(f"交易日范围: {trade_dates[0]} ~ {trade_dates[-1]}")
    
    # 3. 执行转换
    build_event_store(
        df_raw=df_raw,
        trade_dates=trade_dates,
        feature_cols=FEATURE_COLS,
        output_dir=OUTPUT_DIR,
        max_events=MAX_EVENTS,
    )