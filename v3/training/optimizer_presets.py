from __future__ import annotations

from v3.config import merge_dict

OPTIMIZER_PRESETS = {
    "adamw_default": {
        "name": "adamw",
        "optim_params": {"lr": 1e-3, "weight_decay": 1e-4, "eps": 1e-8},
        "accumulation_steps": 1,
        "if_grad_norm": True,
        "max_grad_norm": 3.0,
        "if_lr_decay": True,
        "scheduler": "reduce_lr_on_plateau",
        "sched_params": {"mode": "min", "factor": 0.5, "patience": 4},
        "warmup": {"enabled": False, "name": "linearlr", "epoch": 5, "start_lr": 1e-8},
    },
}


def optimizer_config(name, **override):
    try:
        config = OPTIMIZER_PRESETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown optimizer preset {name!r}; available: {sorted(OPTIMIZER_PRESETS)}") from exc
    return merge_dict(config, override)
