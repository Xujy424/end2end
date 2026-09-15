from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from v3.training.metrics import IC, rankIC
from v3.training.trainer import TRAINER_REGISTRY


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


def run_training(args, model_class, *, framework=None, loss_config=None, run_name=None, **kwargs):
    key = (framework or args.training.get("framework", "supervise")).lower()
    try:
        trainer_fn = TRAINER_REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"Unknown training framework {key!r}; available: {sorted(TRAINER_REGISTRY)}") from exc
    return trainer_fn(args, model_class, loss_config=loss_config, run_name=run_name, **kwargs)


def result_dir(args, *parts) -> Path:
    path = Path(args.training.perf_path).expanduser() / args.model.name / "v3"
    for part in parts:
        path = path / str(part)
    path.mkdir(parents=True, exist_ok=True)
    return path
