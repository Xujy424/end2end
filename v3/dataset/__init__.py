from __future__ import annotations

from v3.dataset.dataloader import *
from v3.training.paths import DATA_ROOT



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
        "feature_blocks": {
         "dailyset": {
                "kind": "daily",
                "data_path": "model_input/dGRU",
                "fields": [
                    "close_zscore", "open_zscore", "high_zscore", "low_zscore", "logvolume_zscore", "turnover_zscore",
                    "close_pct", "open_pct", "high_pct", "low_pct", "logvolume_pct", "turnover_pct",
                    "close2open", "high2open", "low2open", "high2low", "high2close", "low2close",
                ],
                "lag": 20,
            },
            # "minuteset": {
            #     "kind": "minute",
            #     "data_path": "m_essentials",
            #     "fields": ["close2dopen", "high2dopen", "low2dopen", "ppos", "volume_adj2rollmean", "amount2rollmean"],
            # },
        },
    },
}


DATASET_DICT = {
    'batch': BatchDataset,
    'flatten': FlattenDataset,
}


__all__ = [
    "DATASET_DICT",
    "DATASET_PRESETS",
]

