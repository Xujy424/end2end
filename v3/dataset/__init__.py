from __future__ import annotations

from .dataloader import BaseDataset, BatchDataset, FlattenDataset, daily_collate_fn

DATASET_DICT = {
    "datapool_batch": BatchDataset,
    "datapool_flatten": FlattenDataset,
}

multi_collate_fn = daily_collate_fn

__all__ = [
    "DATASET_DICT",
    "BaseDataset",
    "BatchDataset",
    "FlattenDataset",
    "daily_collate_fn",
    "multi_collate_fn",
]
