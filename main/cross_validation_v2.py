from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

from dataset import DATASET_DICT, multi_collate_fn
from training_v2.trainer import SupervisedTrainerV2


def contiguous_kfold_indices(sample_count: int, folds: int = 4):
    """Report-style KFold: hold out each contiguous date block once."""
    if folds < 2 or folds > sample_count:
        raise ValueError("folds must be between 2 and sample_count")
    all_indices = np.arange(sample_count)
    validation_blocks = np.array_split(all_indices, folds)
    return [
        (np.setdiff1d(all_indices, validation, assume_unique=True), validation)
        for validation in validation_blocks
    ]


def cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0).replace(0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def run_kfold_cv(
    args,
    model_class,
    *,
    train_val_range=("2018-01-01", "2021-12-31"),
    prediction_range=("2022-01-01", "2026-03-31"),
    folds=4,
    run_name=None,
):
    """Train K fresh models, then z-score and equal-weight their predictions."""
    fold_predictions = []
    fold_histories = []
    label_df = None
    for fold, (train_indices, valid_indices) in enumerate(
        contiguous_kfold_indices(_sample_count(args, train_val_range), folds), start=1
    ):
        fold_args = copy.deepcopy(args)
        fold_args.training.seed = int(fold_args.training.seed) + fold - 1
        trainer = SupervisedTrainerV2(fold_args, model_class, run_name=f"{run_name or fold_args.model.loss.name}/fold_{fold}")
        dataset = trainer._dataset(*train_val_range)
        ordered = getattr(trainer.loss, "requires_ordered_batches", False)
        workers = int(fold_args.training.get("num_workers", 0))
        loader_options = dict(batch_size=1, num_workers=workers, pin_memory=trainer.device.type == "cuda",
                              drop_last=False, persistent_workers=workers > 0, collate_fn=multi_collate_fn)
        train_loader = DataLoader(Subset(dataset, train_indices.tolist()), shuffle=not ordered, **loader_options)
        valid_loader = DataLoader(Subset(dataset, valid_indices.tolist()), shuffle=False, **loader_options)
        fold_histories.append(trainer.fit(train_loader, valid_loader))
        prediction, label_df = trainer.predict(prediction_range, save=True)
        fold_predictions.append(cross_sectional_zscore(prediction))

    ensemble = sum(fold_predictions) / len(fold_predictions)
    output_dir = Path(args.training.perf_path).expanduser() / args.model.name / "v2" / (run_name or args.model.loss.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensemble.to_csv(output_dir / "alpha_kfold_ensemble.csv")
    label_df.to_csv(output_dir / "label_kfold.csv")
    return ensemble, label_df, fold_histories


def _sample_count(args, date_range):
    params = copy.deepcopy(args.training.dataset.params)
    params.shared_param_dict.start_date = date_range[0]
    params.shared_param_dict.end_date = date_range[1]
    return len(DATASET_DICT[args.training.dataset.name](**params))


class _NoModel:
    """Sentinel replaced before construction; keeps public API model-centric."""

    def __init__(self, **kwargs):
        raise RuntimeError("_NoModel must never be instantiated")
