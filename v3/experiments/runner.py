from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from v3.training.metrics import IC, rankIC
from v3.training.trainer import SupervisedTrainerV3, run_kfold_supervise, run_rolling_supervise


def apply_loss_config(args, loss_config: Mapping[str, Any] | None = None):
    run_args = copy.deepcopy(args)
    if loss_config:
        run_args.model.loss.name = loss_config.get("name", run_args.model.loss.name)
        run_args.model.loss.params = loss_config.get("params", {})
    return run_args


def summarize_prediction(prediction: pd.DataFrame, label: pd.DataFrame) -> dict[str, float]:
    pred_values, label_values = prediction.values, label.values
    rank_ic = rankIC(pred_values, label_values)
    pearson_ic = IC(pred_values, label_values)
    rank_std = np.nanstd(rank_ic)
    ic_std = np.nanstd(pearson_ic)
    return {
        "rank_ic_mean": float(np.nanmean(rank_ic)),
        "rank_ic_ir": float(np.nanmean(rank_ic) / rank_std) if rank_std else np.nan,
        "ic_mean": float(np.nanmean(pearson_ic)),
        "ic_ir": float(np.nanmean(pearson_ic) / ic_std) if ic_std else np.nan,
    }


def run_supervise(args, model_class, *, loss_config=None, prediction_range=None, run_name=None):
    run_args = apply_loss_config(args, loss_config)
    trainer = SupervisedTrainerV3(run_args, model_class, run_name=run_name)
    history = trainer.fit()
    pred, label = trainer.predict(prediction_range, save=True)
    return pred, label, [history]


def run_training(args, model_class, *, framework="supervise", loss_config=None, run_name=None, **kwargs):
    framework = framework.lower()
    if framework in {"supervise", "basic", "single"}:
        return run_supervise(args, model_class, loss_config=loss_config, run_name=run_name, **kwargs)
    if framework in {"rolling", "rolling_supervise"}:
        return run_rolling_supervise(args, model_class, loss_config=loss_config, run_name=run_name, **kwargs)
    if framework in {"kfold", "cv", "cross_validation"}:
        return run_kfold_supervise(args, model_class, loss_config=loss_config, run_name=run_name, **kwargs)
    raise ValueError(f"Unknown framework {framework!r}")


def result_dir(args, *parts) -> Path:
    path = Path(args.training.perf_path).expanduser() / args.model.name / "v3"
    for part in parts:
        path = path / str(part)
    path.mkdir(parents=True, exist_ok=True)
    return path
