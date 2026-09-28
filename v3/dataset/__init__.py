from __future__ import annotations

from v3.dataset.dataloader import *
from v3.training.paths import DATA_ROOT


DAILY_FIELDS = [
    "close_zscore", "open_zscore", "high_zscore", "low_zscore", "logvolume_zscore", "turnover_zscore",
    "close_pct", "open_pct", "high_pct", "low_pct", "logvolume_pct", "turnover_pct",
    "close2open", "high2open", "low2open", "high2low", "high2close", "low2close",
]

MINUTE_FIELDS = [
    "close2dopen", "high2dopen", "low2dopen", "ppos", "volume_adj2rollmean",
]

DAILY_FEATURE_BLOCKS = {
    "dailyset": {
        "kind": "daily",
        "data_path": "model_input/dGRU",
        "fields": DAILY_FIELDS,
        "lag": 20,
    },
}

MINUTE_FEATURE_BLOCKS = {
    "minuteset": {
        "kind": "minute",
        "data_path": "model_input/mGRU",
        "fields": MINUTE_FIELDS,
    },
}

PRIME_FIELDS = {
    "bull": [
        "momentum_5d", "momentum_10d", "momentum_20d", "rsi_6", "rsi_14",
        "north_net_flow", "main_net_inflow", "roe_growth", "revenue_growth",
        "profit_ratio", "price_vs_ma20", "volume_ratio", "chip_support",
        "winner_rate", "clv_20d_avg", "upside_volume_ratio_20d",
    ],
    "bear": [
        "pe_rank", "pb_rank", "bias_20d", "bias_60d", "trapped_ratio",
        "dist_to_resistance", "cost_pressure", "avg_cost_deviation",
        "debt_ratio", "goodwill_risk",
    ],
    "friction": [
        "turnover_rate", "turnover_20d_avg", "volatility_20d", "volatility_60d",
        "amplitude", "amplitude_20d_avg", "asr", "chip_concentration_change",
        "vol_price_divergence",
    ],
    "macro": [
        "cpi_yoy", "ppi_yoy", "pmi", "lpr_1y", "csi500_ret_20d",
        "market_turnover", "market_volatility", "market_liquidity_change",
        "real_rate_proxy",
    ],
}

PRIME_FEATURE_BLOCKS = {
    name: {
        "kind": "daily",
        "data_path": f"model_input/prime/{name}",
        "fields": fields,
        "lag": 1,
    }
    for name, fields in PRIME_FIELDS.items()
}


DATASET_PRESETS = {
    "name": "batch",
    "params": {
        "dataset_config": {
            "root": DATA_ROOT,
            "asset": "stock",
            "label": "Y.10D",
            "mode": "universe",
            "pool_name": None,
            "fix_stock": None,
            "sample_size": None,
            "nan_filter_blocks": ["dailyset"],
        },
        "feature_blocks": DAILY_FEATURE_BLOCKS,
    },
}


DATASET_DICT = {
    'batch': BatchDataset,
    'flatten': FlattenDataset,
}


__all__ = [
    "DATASET_DICT",
    "DATASET_PRESETS",
    "DAILY_FEATURE_BLOCKS",
    "MINUTE_FEATURE_BLOCKS",
    "PRIME_FEATURE_BLOCKS",
    "PRIME_FIELDS",
    "DAILY_FIELDS",
    "MINUTE_FIELDS",
]

