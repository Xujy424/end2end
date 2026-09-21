from __future__ import annotations

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

        stages = list(self.args.training.stages)
        if not stages:
            raise ValueError("training.stages cannot be empty")

        histories = []

        for stage in stages:
            self._configure_stage(stage)

            history = super().fit(
                train_loader=train_loader,
                valid_loader=valid_loader,
                save_loss=False,
                train_range=train_range,
                valid_range=valid_range,
                warm_start_path=warm_start_path,
                num_epoch=stage.num_epoch,
            )
            self._load_model(self.model_path)
            history.insert(0, "stage", stage.name)
            histories.append(history)
            warm_start_path = self.model_path

        result = pd.concat(histories, ignore_index=True)
        if save_loss:
            result.to_csv(self.perf_dir / "loss_history.csv", index=False)
        return result

    def _configure_stage(self, stage):
        train_modules = set(stage.train_modules)
        module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        for name, param in module.named_parameters():
            param.requires_grad = name.split(".", 1)[0] in train_modules

        trainable = [param for param in module.parameters() if param.requires_grad]
        if not trainable:
            raise ValueError(f"Stage {stage.name!r} has no trainable parameters")

        self._frozen_modules = [
            child
            for name, child in module.named_children()
            if name not in train_modules
        ]
        optimizer_config = to_config(merge_dict(self.args.optimizer, stage.get("optimizer")))
        self.optimizer, self.scheduler = build_optimizer_bundle(optimizer_config, trainable)
        print(f"Stage {stage.name}: train {sorted(train_modules)}")

    def _set_model_mode(self, training):
        super()._set_model_mode(training)
        if training:
            for module in self._frozen_modules:
                module.eval()
