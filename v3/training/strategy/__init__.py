"""Training strategy modules.

Import concrete strategies from their modules, for example
``v3.training.strategy.plain`` or ``v3.training.strategy.kfold``.
The package init intentionally avoids eager imports so a plain run only loads
the plain strategy.
"""

from __future__ import annotations

from importlib import import_module

_STRATEGY_MODULES = {
    "plain": ("v3.training.strategy.plain", "run_plain"),
    "rolling": ("v3.training.strategy.rolling", "run_rolling"),
    "kfold": ("v3.training.strategy.kfold", "run_kfold"),
    "bagging": ("v3.training.strategy.bagging", "run_bagging"),
    "gridsearch": ("v3.training.strategy.gridsearch", "run_gridsearch"),
}

STRATEGY_PRESETS = {
    "plain": {
        "name": "plain",
        "params": {
            "train_range": ("2016-01-01", "2023-12-31"),
            "valid_range": ("2024-01-01", "2024-12-31"),
            "test_range": ("2025-01-01", "2025-12-31"),
        },
    },
    "rolling": {
        "name": "rolling",
        "params": {
            "window_params": {
                "start_dt": "2016-01-01",
                "end_dt": "2025-12-31",
                "train_len": 7,
                "valid_len": 1,
                "test_len": 1,
                "rolling_gap": 1,
            },
        },
    },
    "kfold": {
        "name": "kfold",
        "params": {
            "train_val_range": ("2016-01-01", "2024-12-31"),
            "prediction_range": ("2025-01-01", "2025-12-31"),
            "folds": 5,
        },
    },
}


def get_strategy(name):
    key = str(name).lower()
    try:
        module_name, function_name = _STRATEGY_MODULES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown strategy {name!r}; available: {sorted(_STRATEGY_MODULES)}") from exc
    return getattr(import_module(module_name), function_name)


def strategy_config(name, **params):
    try:
        config = STRATEGY_PRESETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown strategy preset {name!r}; available: {sorted(STRATEGY_PRESETS)}") from exc
    from v3.config import merge_dict

    return merge_dict(config, {"params": params} if params else None)


__all__ = ["STRATEGY_PRESETS", "get_strategy", "strategy_config"]
