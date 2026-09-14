"""GRU V2 training entry point.

Examples:
    python gru_rankic_v2.py --framework kfold --loss rankic --folds 5
    python gru_rankic_v2.py --framework rolling --loss domain_industry
    python gru_rankic_v2.py --ensemble bagging --members 5 --framework kfold
    python gru_rankic_v2.py --ensemble gridsearch --grid temperature=0.005,0.01,0.02 --loss rankic
"""

from __future__ import annotations

import argparse
from pathlib import Path

from model_hub.RNNs.gru import GRU_Arg, GRU_Model
from v2.framework import run_bagging, run_gridsearch, run_training

ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else Path("/data/shanghai/xujiayi/workflow/data/")


LOSS_PRESETS = {
    "mse": {"name": "mse", "params": {}},
    "ic": {"name": "ic", "params": {}},
    "pearson_ic": {"name": "pearson_ic", "params": {}},
    "rankic": {"name": "rankic", "params": {"temperature": 0.01, "method": "sigmoid"}},
    "rankic_neural": {"name": "rankic", "params": {"temperature": 0.01, "method": "neural"}},
    "temporal_rankic": {"name": "temporal_rankic", "params": {"temperature": 0.01, "method": "sigmoid", "turnover_rate": 0.1}},
    "domain_industry": {
        "name": "domain_rankic",
        "params": {
            "temperature": 0.01,
            "method": "sigmoid",
            "domain_type": "industry",
            "provider_params": {"axis_root": ROOT / "axis", "mask_root": ROOT / "mask"},
        },
    },
    "domain_index": {
        "name": "domain_rankic",
        "params": {
            "temperature": 0.01,
            "method": "sigmoid",
            "domain_type": "index",
            "domains": ["zz800", "zz1000", "others"],
            "domain_weights": [0.025, 0.8, 0.175],
            "provider_params": {"axis_root": ROOT / "axis", "mask_root": ROOT / "mask"},
        },
    },
}


def parse_grid(items):
    grid = {}
    for item in items or []:
        key, raw_values = item.split("=", 1)
        values = []
        for value in raw_values.split(","):
            try:
                values.append(float(value))
            except ValueError:
                values.append(value)
        grid[key] = values
    return grid


def build_args(config_override=None):
    return GRU_Arg(config_override)


def run_gru_v2(config_override=None, *, framework="supervise", loss="rankic", ensemble="none", members=5, folds=5, grid=None, **kwargs):
    args = build_args(config_override)
    loss_config = LOSS_PRESETS[loss] if isinstance(loss, str) else loss
    if framework in {"kfold", "cv", "cross_validation"}:
        kwargs.setdefault("folds", folds)
    if ensemble == "bagging":
        return run_bagging(args, GRU_Model, members=members, framework=framework, loss_config=loss_config, **kwargs)
    if ensemble == "gridsearch":
        return run_gridsearch(args, GRU_Model, grid or {}, framework=framework, base_loss_config=loss_config, **kwargs)
    return run_training(args, GRU_Model, framework=framework, loss_config=loss_config, run_name=loss, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description="GRU V2 flexible training entry point")
    parser.add_argument("--framework", choices=("supervise", "rolling", "kfold"), default="supervise")
    parser.add_argument("--loss", choices=tuple(LOSS_PRESETS), default="rankic")
    parser.add_argument("--ensemble", choices=("none", "bagging", "gridsearch"), default="none")
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--train-start")
    parser.add_argument("--train-end")
    parser.add_argument("--valid-start")
    parser.add_argument("--valid-end")
    parser.add_argument("--test-start")
    parser.add_argument("--test-end")
    parser.add_argument("--grid", action="append", help="grid item like temperature=0.005,0.01")
    parsed = parser.parse_args(argv)

    period_override = {}
    for key in ("train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end"):
        value = getattr(parsed, key)
        if value:
            period_override[key] = value
    config_override = {"training": {"period": period_override}} if period_override else None
    return run_gru_v2(
        config_override,
        framework=parsed.framework,
        loss=parsed.loss,
        ensemble=parsed.ensemble,
        members=parsed.members,
        folds=parsed.folds,
        grid=parse_grid(parsed.grid),
    )


if __name__ == "__main__":
    main()
