from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from main.cross_validation_v2 import cross_sectional_zscore
from training_v2.trainer import SupervisedTrainerV2


def run_bagging(args, model_class, *, members=5, prediction_range=None, standardize=True):
    """Callable seed-bagging wrapper for the V2 trainer."""
    if members < 1:
        raise ValueError("members must be positive")
    predictions = []
    label_df = None
    histories = []
    for member in range(1, members + 1):
        member_args = copy.deepcopy(args)
        member_args.training.seed = int(member_args.training.seed) + member - 1
        trainer = SupervisedTrainerV2(member_args, model_class, run_name=f"bagging/member_{member}")
        histories.append(trainer.fit())
        prediction, label_df = trainer.predict(prediction_range, save=True)
        predictions.append(cross_sectional_zscore(prediction) if standardize else prediction)
    ensemble = sum(predictions) / len(predictions)
    output_dir = Path(args.training.perf_path).expanduser() / args.model.name / "v2" / "bagging"
    output_dir.mkdir(parents=True, exist_ok=True)
    ensemble.to_csv(output_dir / "alpha_ensemble.csv")
    label_df.to_csv(output_dir / "label.csv")
    return ensemble, label_df, histories
