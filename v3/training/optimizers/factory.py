from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


OPTIMIZERS = {
    "adam": torch.optim.Adam, 
    "adamw": torch.optim.AdamW
}

SCHEDULERS = {
    "linearlr": torch.optim.lr_scheduler.LinearLR,
    "steplr": torch.optim.lr_scheduler.StepLR,
    "multi_steplr": torch.optim.lr_scheduler.MultiStepLR,
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    "reduce_lr_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
}


def build_optimizer(name, parameters, params=None):
    try:
        optimizer_class = OPTIMIZERS[name.lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown optimizer {name!r}; available: {sorted(OPTIMIZERS)}") from exc
    return optimizer_class(parameters, **dict(params or {}))


def build_scheduler(name, optimizer, params=None):
    try:
        scheduler_class = SCHEDULERS[name.lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown scheduler {name!r}; available: {sorted(SCHEDULERS)}") from exc
    return scheduler_class(optimizer, **dict(params or {}))


class EarlyStopping:
    def __init__(self, patience=7, delta=0.0, best_loss=np.inf):
        self.patience = int(patience)
        self.delta = float(delta)
        self.best_loss = float(best_loss)
        self.counter = 0
        self.early_stop = False

    def __call__(self, value, model, path):
        if value < self.best_loss - self.delta:
            self.best_loss = float(value)
            self.counter = 0
            module = model.module if isinstance(model, nn.DataParallel) else model
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(module.state_dict(), path)
        else:
            self.counter += 1
            self.early_stop = self.counter >= self.patience
