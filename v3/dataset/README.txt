"dataset": {
    "name": "datapool_batch",
    "params": {
        "dataset_config": {...},
        "feature_blocks": {...},
    },
}

"datapool_batch"    # 一天一个样本/一个截面
    dailyset  -> (stock, lag, feature)
    minuteset -> (stock, minute, feature)
    label     -> (stock,)

"datapool_flatten"  # 一天一只股票一个样本
    dailyset  -> (lag, feature)
    minuteset -> (minute, feature)
    label     -> (1,)



"dataset_config": {
    "root": "Z:/",              # 可选，DataPool 根目录；默认自动 ROOT
    "asset": "stock",           # 可选，默认 stock
    "label": "Y.10D",           # 标签字段，会映射到 stock/model_input/labels/Y.10D.bin
    "mode": "universe",         # 股票池模式
    "pool_name": None,          # mode="pool" 时使用，如 "zz800"
    "fix_stock": None,          # mode="fix" 时指定股票列表
    "sample_size": None,        # mode="sample" 时每日随机抽样数量
    "nan_filter_blocks": ["dailyset"] # 用哪些 feature block 做 NaN 过滤
}

"universe"  # 全市场可交易且 label 非空股票
"pool"      # 在 universe 基础上叠加 mask/{pool_name}_mask.bin
"fix"       # 只取 fix_stock 指定股票
"sample"    # 从 universe 里每日随机抽 sample_size 只



"feature_blocks": {
    "dailyset": {
        "kind": "daily",
        "data_path": "model_input/dGRU",
        "fields": ["close_zscore", "close_pct"],
        "lag": 20,
    },
    "minuteset": {
        "kind": "minute",
        "data_path": "m_essentials",
        "fields": ["close", "volume"],
    },
}

"kind"       # daily 或 minute；不写时根据名字推断 dailyset/minuteset
"data_path"  # stock 下的相对目录
"fields"     # 字段名列表
"lag"        # 只对 daily 有意义，表示过去多少天窗口
