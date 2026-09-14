from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from main.cross_validation_v2 import run_kfold_cv
from training.metrics import IC, rankIC

ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else Path('/data/shanghai/xujiayi/workflow/data/')



DEFAULT_LOSSES = {
    "mse": {},
    "pearson_ic": {},
    "rankic_sigmoid": {"name": "rankic", "params": {"temperature": 0.01, "method": "sigmoid"}},
    "rankic_neural": {"name": "rankic", "params": {"temperature": 0.01, "method": "neural"}},
    "domain_rankic": {
        "name": "domain_rankic",
        "params": {
            "temperature": 0.01,
            "domain_type": "index",
            "domains": ["zz800", "zz1000", "others"],
            "domain_weights": [0.025, 0.8, 0.175],
            "axis_root": ROOT/"axis",
            "mask_root": ROOT/"stock/index/mask",
        },
    },
    "industry_rankic": {
        "name": "domain_rankic",
        "params": {
            "temperature": 0.01,
            "domain_type": "industry",
            "provider_params": {
                "axis_root": ROOT/"axis",
                "mask_root": ROOT/"stock/mask",
            },
        },
    },
    "temporal_rankic": {"name": "temporal_rankic", "params": {"temperature": 0.01, "turnover_rate": 0.1}},
}


def summarize_prediction(prediction: pd.DataFrame, label: pd.DataFrame):
    pred_values, label_values = prediction.values, label.values
    rank_ic = rankIC(pred_values, label_values)
    pearson_ic = IC(pred_values, label_values)
    return {
        "rank_ic_mean": float(np.nanmean(rank_ic)),
        "rank_ic_ir": float(np.nanmean(rank_ic) / np.nanstd(rank_ic)) if np.nanstd(rank_ic) else np.nan,
        "ic_mean": float(np.nanmean(pearson_ic)),
        "ic_ir": float(np.nanmean(pearson_ic) / np.nanstd(pearson_ic)) if np.nanstd(pearson_ic) else np.nan,
    }


def run_loss_comparison(
    args,
    model_class,
    *,
    loss_configs=None,
    train_val_range=("2018-01-01", "2021-12-31"),
    prediction_range=("2022-01-01", "2026-03-31"),
    folds=4,
):
    """Callable experiment wrapper; importing this module never starts training."""
    configs = loss_configs or DEFAULT_LOSSES
    rows = []
    outputs = {}
    for alias, config in configs.items():
        experiment_args = copy.deepcopy(args)
        experiment_args.model.loss.name = config.get("name", alias)
        experiment_args.model.loss.params = config.get("params", {})
        prediction, label, histories = run_kfold_cv(
            experiment_args,
            model_class,
            train_val_range=train_val_range,
            prediction_range=prediction_range,
            folds=folds,
            run_name=alias,
        )
        rows.append({"loss": alias, **summarize_prediction(prediction, label)})
        outputs[alias] = {"prediction": prediction, "label": label, "histories": histories}
    summary = pd.DataFrame(rows).set_index("loss").sort_values("rank_ic_mean", ascending=False)
    output_dir = Path(args.training.perf_path).expanduser() / args.model.name / "v2" / "loss_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv")
    return summary, outputs
