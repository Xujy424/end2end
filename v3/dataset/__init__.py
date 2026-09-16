from __future__ import annotations

from importlib import import_module

from v3.paths import DATA_ROOT

DATASET_PRESETS = {
    "gru_daily": {
        "name": "datapool_batch",
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
            "feature_blocks": {},
        },
    },
}


class _DatasetDict(dict):
    def __getitem__(self, key):
        if key == "datapool_batch":
            from .dataloader import BatchDataset

            return BatchDataset
        if key == "datapool_flatten":
            from .dataloader import FlattenDataset

            return FlattenDataset
        raise KeyError(key)

    def keys(self):
        return {"datapool_batch": None, "datapool_flatten": None}.keys()


DATASET_DICT = _DatasetDict()


def dataset_config(name, **params):
    try:
        config = DATASET_PRESETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset preset {name!r}; available: {sorted(DATASET_PRESETS)}") from exc
    from v3.config import merge_dict

    return merge_dict(config, params)

__all__ = [
    "DATASET_DICT",
    "DATASET_PRESETS",
    "dataset_config",
]


def __getattr__(name):
    if name in {"BaseDataset", "BatchDataset", "FlattenDataset"}:
        dataloader = import_module("v3.dataset.dataloader")

        return getattr(dataloader, name)
    raise AttributeError(name)
