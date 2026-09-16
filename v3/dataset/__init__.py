from __future__ import annotations

from importlib import import_module

__all__ = [
    "DATASET_DICT",
    "BaseDataset",
    "BatchDataset",
    "FlattenDataset",
    "daily_collate_fn",
    "multi_collate_fn",
]


def __getattr__(name):
    if name in __all__:
        dataloader = import_module("v3.dataset.dataloader")

        if name == "DATASET_DICT":
            return {
                "datapool_batch": dataloader.BatchDataset,
                "datapool_flatten": dataloader.FlattenDataset,
            }
        if name == "multi_collate_fn":
            return dataloader.daily_collate_fn
        return getattr(dataloader, name)
    raise AttributeError(name)
