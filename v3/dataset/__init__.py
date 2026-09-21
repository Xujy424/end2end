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
    "DAILY_FIELDS",
    "MINUTE_FIELDS",
]

