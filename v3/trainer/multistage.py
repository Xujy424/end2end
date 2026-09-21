from __future__ import annotations

from pathlib import Path
import copy
import re

import pandas as pd
from torch import nn

from v3.config import merge_dict, to_config
from v3.optimizers import build_optimizer_bundle
from v3.trainer.supervised import SupervisedTrainerV3


class MultiStageSupervisedTrainerV3(SupervisedTrainerV3):
    """Run multiple parameter-freezing stages inside one training window."""

    def __init__(self, args, model_class, loss=None, run_name=None):
        self._frozen_modules = []
        super().__init__(args, model_class, loss=loss, run_name=run_name)

    def fit(
        self,
        train_loader=None,
        valid_loader=None,
        save_loss=True,
        train_range=None,
        valid_range=None,
        warm_start_path=None,
    ):
        if train_range is None or valid_range is None:
            raise ValueError("fit requires explicit train_range and valid_range")

        ordered = getattr(self.loss, "requires_ordered_batches", False)
        train_loader = train_loader or self.make_loader(self.make_dataset(train_range), shuffle=not ordered)
        valid_loader = valid_loader or self.make_loader(self.make_dataset(valid_range))

        stages = self._stage_configs()
        final_model_path = self.model_path
        original_optimizer = copy.deepcopy(self.args.optimizer)
        original_num_epoch = int(self.args.training.num_epoch)
        incoming_path = warm_start_path or self._initial_model_path()
        histories = []

        try:
            for index, stage in enumerate(stages, start=1):
                stage_name = str(stage.get("name", f"stage_{index:02d}"))
                self._configure_stage(stage)
                self._configure_stage_optimizer(stage, original_optimizer)
                self.args.training.num_epoch = int(stage.get("num_epoch", original_num_epoch))

                stage_path = self.perf_dir / "stages" / f"{index:02d}_{_safe_name(stage_name)}" / "best_model.pth"
                self.model_path = stage_path
                history = super().fit(
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    save_loss=False,
                    train_range=train_range,
                    valid_range=valid_range,
                    warm_start_path=incoming_path,
                )
                self._load_model(stage_path)
                history.insert(0, "stage", stage_name)
                history.insert(1, "stage_index", index)
                histories.append(history)
                incoming_path = stage_path
        finally:
            self.args.optimizer = original_optimizer
            self.args.training.num_epoch = original_num_epoch
            self.model_path = final_model_path

        self._save_model(final_model_path)
        result = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
        if save_loss:
            result.to_csv(self.perf_dir / "loss_history.csv", index=False)
        return result

    def _stage_configs(self):
        stages = self.args.training.get("stages", None)
        if stages:
            return list(stages)

        transfer = self.args.training.get("transfer", None)
        if transfer:
            stage = dict(transfer)
            stage.setdefault("name", "transfer")
            return [stage]
        raise ValueError("Multi-stage training requires training.stages")

    def _initial_model_path(self):
        multistage = self.args.training.get("multistage", {}) or {}
        transfer = self.args.training.get("transfer", {}) or {}
        checkpoint = multistage.get("initial_model_path") or transfer.get("pretrained_path")
        return Path(checkpoint) if checkpoint else None

    def _configure_stage(self, stage):
        freeze_patterns = tuple(stage.get("freeze_patterns", ()))
        train_patterns = tuple(stage.get("train_patterns", ()))
        module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        for name, param in module.named_parameters():
            param.requires_grad = not train_patterns or _matches(name, train_patterns)
            if _matches(name, freeze_patterns):
                param.requires_grad = False

        trainable = [name for name, param in module.named_parameters() if param.requires_grad]
        frozen = [name for name, param in module.named_parameters() if not param.requires_grad]
        if not trainable:
            raise ValueError(f"Stage {stage.get('name', '<unnamed>')!r} has no trainable parameters")

        self._frozen_modules = [
            child
            for name, child in module.named_modules()
            if name
            and any(True for _ in child.parameters(recurse=True))
            and all(not param.requires_grad for param in child.parameters(recurse=True))
        ]
        print(f"Stage {stage.get('name', '<unnamed>')} trainable params: {trainable}")
        print(f"Stage {stage.get('name', '<unnamed>')} frozen params: {frozen}")

    def _configure_stage_optimizer(self, stage, base_optimizer):
        optimizer_config = merge_dict(base_optimizer, stage.get("optimizer", None))
        self.args.optimizer = to_config(optimizer_config)
        trainable = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer, self.scheduler = build_optimizer_bundle(self.args.optimizer, trainable)

    def _set_model_mode(self, training):
        super()._set_model_mode(training)
        if training:
            for module in self._frozen_modules:
                module.eval()


def _matches(name, patterns):
    return any(pattern == "*" or pattern in name for pattern in patterns)


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "stage"
