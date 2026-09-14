from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from v3.experiments import run_bagging, run_gridsearch, run_training
from v3.models.gru import GRUConfig, GRUModel

ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else Path("/data/shanghai/xujiayi/workflow/data/")

LOSS_PRESETS = {
    "mse": {"name": "mse", "params": {}},
    "ic": {"name": "ic", "params": {}},
    "pearson_ic": {"name": "pearson_ic", "params": {}},
    "rankic": {"name": "rankic", "params": {"temperature": 0.01, "method": "sigmoid"}},
    "rankic_neural": {"name": "rankic", "params": {"temperature": 0.01, "method": "neural"}},
    "temporal_rankic": {
        "name": "temporal_rankic",
        "params": {"temperature": 0.01, "method": "sigmoid", "turnover_rate": 0.1},
    },
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


def loss_config(name_or_config="rankic", **params):
    if isinstance(name_or_config, Mapping):
        config = deepcopy(dict(name_or_config))
    else:
        config = deepcopy(LOSS_PRESETS[name_or_config])
    config.setdefault("params", {}).update(params)
    return config


def period_config(
    *,
    train_start=None,
    train_end=None,
    valid_start=None,
    valid_end=None,
    test_start=None,
    test_end=None,
):
    values = {
        key: value
        for key, value in {
            "train_start": train_start,
            "train_end": train_end,
            "valid_start": valid_start,
            "valid_end": valid_end,
            "test_start": test_start,
            "test_end": test_end,
        }.items()
        if value is not None
    }
    return {"training": {"period": values}} if values else {}


def run_gru(
    *,
    framework="supervise",
    loss="rankic",
    ensemble="none",
    config_override: Mapping[str, Any] | None = None,
    loss_params: Mapping[str, Any] | None = None,
    members=5,
    folds=5,
    grid: Mapping[str, list[Any]] | None = None,
    **kwargs,
):
    args = GRUConfig(config_override)
    selected_loss = loss_config(loss, **dict(loss_params or {}))
    if framework in {"kfold", "cv", "cross_validation"}:
        kwargs.setdefault("folds", folds)
    if ensemble == "bagging":
        return run_bagging(args, GRUModel, members=members, framework=framework, loss_config=selected_loss, **kwargs)
    if ensemble == "gridsearch":
        return run_gridsearch(args, GRUModel, grid or {}, framework=framework, base_loss_config=selected_loss, **kwargs)
    return run_training(args, GRUModel, framework=framework, loss_config=selected_loss, run_name=selected_loss["name"], **kwargs)


def run_rankic_kfold_5():
    return run_gru(framework="kfold", loss="rankic", folds=5)


def run_domain_rolling():
    return run_gru(framework="rolling", loss="domain_industry")


def run_rankic_gridsearch():
    return run_gru(
        framework="kfold",
        loss="rankic",
        ensemble="gridsearch",
        folds=5,
        grid={"temperature": [0.005, 0.01, 0.02], "method": ["sigmoid", "neural"]},
    )


if __name__ == "__main__":
    # Keep this file runnable for smoke usage, but production runs should call
    # run_gru(...) with explicit keyword arguments from notebooks or job scripts.
    run_rankic_kfold_5()
