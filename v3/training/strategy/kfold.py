from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from v3.training.trainer import resolve_trainer_class


def contiguous_kfold_indices(sample_count: int, folds: int = 5):
    if folds < 2 or folds > sample_count:
        raise ValueError("folds must be between 2 and sample_count")
    all_indices = np.arange(sample_count)
    valid_blocks = np.array_split(all_indices, folds)
    return [(np.setdiff1d(all_indices, block, assume_unique=True), block) for block in valid_blocks]


def run_kfold(
    args,
    model_class,
    *,
    trainer="supervised",
    trainer_class=None,
    train_val_range=None,
    prediction_range=None,
    folds=5,
    run_name=None,
    standardize=True,
):
    trainer_class = resolve_trainer_class(trainer, trainer_class)
    if train_val_range is None or prediction_range is None:
        raise ValueError("run_kfold requires train_val_range and prediction_range")
    predictions, histories, label_df = [], [], None
    base_name = run_name or args.loss.name
    sample_count = _sample_count(args, model_class, trainer_class, train_val_range)
    for fold, (train_idx, valid_idx) in enumerate(contiguous_kfold_indices(sample_count, folds), start=1):
        fold_args = copy.deepcopy(args)
        fold_args.training.seed = int(fold_args.training.seed) + fold - 1
        trainer = trainer_class(fold_args, model_class, run_name=f"{base_name}/kfold/fold_{fold:02d}")
        dataset = trainer.make_dataset(train_val_range)
        ordered = getattr(trainer.loss, "requires_ordered_batches", False)
        train_loader = trainer.make_loader(dataset, indices=train_idx, shuffle=not ordered)
        valid_loader = trainer.make_loader(dataset, indices=valid_idx)
        histories.append(
            trainer.fit(
                train_loader=train_loader,
                valid_loader=valid_loader,
                train_range=train_val_range,
                valid_range=train_val_range,
            )
        )
        pred, label_df = trainer.predict(prediction_range, save=True)
        predictions.append(_cross_sectional_zscore(pred) if standardize else pred)
    ensemble = sum(predictions) / len(predictions)
    out_dir = Path(args.training.perf_path).expanduser() / args.model.name / "v3" / base_name / "kfold"
    out_dir.mkdir(parents=True, exist_ok=True)
    ensemble.to_csv(out_dir / "alpha_kfold_ensemble.csv")
    label_df.to_csv(out_dir / "label_kfold.csv")
    return ensemble, label_df, histories


def _sample_count(args, model_class, trainer_class, date_range):
    sample_args = copy.deepcopy(args)
    trainer = trainer_class(sample_args, model_class, run_name="_sample_count")
    return len(trainer.make_dataset(date_range))


def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0).replace(0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)
