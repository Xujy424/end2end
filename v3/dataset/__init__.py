from __future__ import annotations

from .dataloader import BaseDataset, BatchDataset, FlattenDataset

DATASET_DICT = {
    "datapool_batch": BatchDataset,
    "datapool_flatten": FlattenDataset,
}

__all__ = [
    "DATASET_DICT",
    "BaseDataset",
    "BatchDataset",
    "FlattenDataset",
]
