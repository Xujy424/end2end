from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

OPTIMIZER_REGISTRY = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}

SCHEDULER_REGISTRY = {
    "linearlr": torch.optim.lr_scheduler.LinearLR,
    "steplr": torch.optim.lr_scheduler.StepLR,
    "multi_steplr": torch.optim.lr_scheduler.MultiStepLR,
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    "reduce_lr_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
}


def build_optimizer(name, parameters, params=None):
    try:
        optimizer_class = OPTIMIZER_REGISTRY[name.lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown optimizer {name!r}; available: {sorted(OPTIMIZER_REGISTRY)}") from exc
    return optimizer_class(parameters, **dict(params or {}))


def build_scheduler(name, optimizer, params=None):
    try:
        scheduler_class = SCHEDULER_REGISTRY[name.lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown scheduler {name!r}; available: {sorted(SCHEDULER_REGISTRY)}") from exc
    return scheduler_class(optimizer, **dict(params or {}))


def build_optimizer_bundle(config, parameters):
    optimizer = build_optimizer(config.name, parameters, config.get("optim_params", {}))
    scheduler = None
    warmup = config.get("warmup", {}) or {}
    use_main = bool(config.get("if_lr_decay", False))
    use_warmup = bool(warmup.get("enabled", False))

    if use_main:
        scheduler = build_scheduler(config.scheduler, optimizer, config.get("sched_params", {}))
    if use_warmup:
        warmup_epochs = int(warmup.get("epoch", warmup.get("epochs", 1)))
        warmup_params = warmup.get("params", {
            "start_factor": float(warmup.get("start_factor", 1e-3)),
            "total_iters": warmup_epochs,
        })
        warmup_scheduler = build_scheduler(warmup.get("name", "linearlr"), optimizer, warmup_params)
        if scheduler is None:
            scheduler = warmup_scheduler
        elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler = WarmupThenPlateau(warmup_scheduler, scheduler, warmup_epochs)
        else:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[warmup_epochs],
            )
    return optimizer, scheduler


class WarmupThenPlateau:
    def __init__(self, warmup_scheduler, plateau_scheduler, warmup_epochs):
        self.warmup_scheduler = warmup_scheduler
        self.plateau_scheduler = plateau_scheduler
        self.warmup_epochs = int(warmup_epochs)
        self.epoch = 0

    def step(self, metric=None):
        self.epoch += 1
        if self.epoch <= self.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            self.plateau_scheduler.step(metric)


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


__all__ = [
    "EarlyStopping",
    "OPTIMIZER_REGISTRY",
    "SCHEDULER_REGISTRY",
    "WarmupThenPlateau",
    "build_optimizer",
    "build_optimizer_bundle",
    "build_scheduler",
]
