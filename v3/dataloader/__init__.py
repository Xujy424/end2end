from typing import Dict, List, Any
import torch as th
import numpy as np

from .multicompose import MultiBatchDataset, MultiFlattenDataset


DATASET_DICT = {
    'batch': MultiBatchDataset,
    "flatten": MultiFlattenDataset,
}


def multi_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:

    feats_collated = {
        key: th.cat([
            sample['feats'][key].unsqueeze(0) if (sample['feats'][key].dim()==2) and (key in ['dailyset','minuteset']) else sample['feats'][key]
            for sample in batch
            ],dim=0)
        for key in batch[0]['feats']
    }

    labels = [sample['label'].reshape(-1) for sample in batch]
    label_collated = th.cat(labels)

    date_idx_collated = np.array([sample['date_idx'] for sample in batch])

    tick_idxs_samples = [np.asarray(sample['tick_idxs']).reshape(-1) for sample in batch]
    tick_idxs_collated = np.concatenate(tick_idxs_samples)

    return {
        'feats': feats_collated,
        'label': label_collated,
        'date_idx': date_idx_collated,
        'tick_idxs': tick_idxs_collated,
    }