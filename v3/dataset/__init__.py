from typing import Any, Dict, List

import numpy as np
import torch as th

from .dataloader import (
    BaseDataset,
    BatchDataset,
    FlattenDataset,
)


def daily_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(batch) == 1:
        return batch[0]
    feats = {
        key: th.cat([
            sample["feats"][key].unsqueeze(0) if sample["feats"][key].dim() == 2 else sample["feats"][key]
            for sample in batch
        ], dim=0)
        for key in batch[0]["feats"]
    }
    return {
        "feats": feats,
        "label": th.cat([sample["label"].reshape(-1) for sample in batch], dim=0),
        "date_idx": np.asarray([sample["date_idx"] for sample in batch]),
        "tick_idxs": np.concatenate([np.asarray(sample["tick_idxs"]).reshape(-1) for sample in batch]),
    }


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
