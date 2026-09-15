from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

from v3.dataset import DATASET_DICT, multi_collate_fn
from v3.training.trainer.supervised import SupervisedTrainerV3


def contiguous_kfold_indices(sample_count: int, folds: int = 5):
    if folds < 2 or folds > sample_count:
        raise ValueError("folds must be between 2 and sample_count")
    all_indices = np.arange(sample_count)
    valid_blocks = np.array_split(all_indices, folds)
    return [(np.setdiff1d(all_indices, block, assume_unique=True), block) for block in valid_blocks]


def cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0).replace(0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def run_kfold_supervise(args, model_class, *, train_val_range=None, prediction_range=None, folds=5, loss_config=None, run_name=None, standardize=True):
    if train_val_range is None or prediction_range is None:
        raise ValueError("run_kfold_supervise requires train_val_range and prediction_range")
    sample_count = _sample_count(args, train_val_range)
    predictions, histories, label_df = [], [], None
    base_name = run_name or args.model.loss.name
    for fold, (train_idx, valid_idx) in enumerate(contiguous_kfold_indices(sample_count, folds), start=1):
        fold_args = copy.deepcopy(args)
        fold_args.training.seed = int(fold_args.training.seed) + fold - 1
        if loss_config:
            fold_args.model.loss.name = loss_config.get("name", fold_args.model.loss.name)
            fold_args.model.loss.params = loss_config.get("params", {})
        trainer = SupervisedTrainerV3(fold_args, model_class, run_name=f"{base_name}/kfold/fold_{fold:02d}")
        dataset = trainer._dataset(*train_val_range)
        ordered = getattr(trainer.loss, "requires_ordered_batches", False)
        workers = int(fold_args.training.get("num_workers", 0))
        opts = dict(
            batch_size=1,
            num_workers=workers,
            pin_memory=bool(fold_args.training.get("pin_memory", trainer.device.type == "cuda")),
            drop_last=False,
            persistent_workers=bool(fold_args.training.get("persistent_workers", workers > 0)) and workers > 0,
            collate_fn=multi_collate_fn,
        )
        if workers > 0:
            opts["prefetch_factor"] = int(fold_args.training.get("prefetch_factor", 4))
        train_loader = DataLoader(Subset(dataset, train_idx.tolist()), shuffle=not ordered, **opts)
        valid_loader = DataLoader(Subset(dataset, valid_idx.tolist()), shuffle=False, **opts)
        histories.append(trainer.fit(train_loader, valid_loader))
        pred, label_df = trainer.predict(prediction_range, save=True)
        predictions.append(cross_sectional_zscore(pred) if standardize else pred)
    ensemble = sum(predictions) / len(predictions)
    out_dir = Path(args.training.perf_path).expanduser() / args.model.name / "v3" / base_name / "kfold"
    out_dir.mkdir(parents=True, exist_ok=True)
    ensemble.to_csv(out_dir / "alpha_kfold_ensemble.csv")
    label_df.to_csv(out_dir / "label_kfold.csv")
    return ensemble, label_df, histories


def _sample_count(args, date_range):
    params = copy.deepcopy(args.training.dataset.params)
    return len(DATASET_DICT[args.training.dataset.name](start_date=date_range[0], end_date=date_range[1], **params))



