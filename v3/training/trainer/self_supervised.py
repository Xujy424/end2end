from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from v3.dataset import DATASET_DICT, multi_collate_fn
from v3.training.losses import build_loss
from v3.training.optimizers import EarlyStopping, build_optimizer_bundle


class BasicSelfSupervisedTrainerV3:
    """Basic self-supervised trainer.

    Models may return their own loss as ``(representation, loss)`` or
    ``{"prediction": ..., "loss": ...}``. Alternatively, configure a loss and
    provide ``args.model.loss.params.target_key``.
    """

    def __init__(self, args, model_class, loss=None, run_name=None):
        self.args = copy.deepcopy(args)
        self.device = torch.device(self.args.training.device if torch.cuda.is_available() else "cpu")
        self._set_seed(int(self.args.training.seed))

        self.model = model_class(**self.args.model.params).to(self.device).float()
        if torch.cuda.device_count() > 1 and self.args.training.multi_gpu:
            self.model = nn.DataParallel(
                self.model,
                device_ids=list(self.args.training.available_gpu),
                output_device=int(self.args.training.main_gpu),
            )

        loss_name = str(self.args.model.loss.get("name", "model")).lower()
        loss_params = dict(self.args.model.loss.get("params", {}))
        self.target_key = loss_params.pop("target_key", None)
        self.prediction_key = loss_params.pop("prediction_key", "prediction")
        self.representation_key = loss_params.pop("representation_key", "representation")
        self.loss = None if loss_name in {"model", "self_supervised"} else (loss or build_loss(loss_name, loss_params))

        self.optimizer, self.scheduler = build_optimizer_bundle(
            self.args.optimizer,
            (p for p in self.model.parameters() if p.requires_grad),
        )

        suffix = run_name or self.args.model.loss.name
        self.perf_dir = Path(self.args.training.perf_path).expanduser() / self.args.model.name / "v3" / suffix
        self.perf_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.perf_dir / "best_model.pth"

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def make_dataset(self, date_range):
        params = copy.deepcopy(self.args.training.dataset.params)
        params.setdefault("dataset_config", {})
        params["dataset_config"]["label"] = None
        return DATASET_DICT[self.args.training.dataset.name](
            start_date=date_range[0],
            end_date=date_range[1],
            **params,
        )

    def make_loader(self, dataset, *, shuffle=False, indices=None):
        if indices is not None:
            from torch.utils.data import Subset

            dataset = Subset(dataset, list(indices))
        workers = int(self.args.training.get("num_workers", 0))
        loader_kwargs = {
            "batch_size": int(self.args.training.get("batch_size", 1)),
            "shuffle": shuffle,
            "num_workers": workers,
            "pin_memory": bool(self.args.training.get("pin_memory", self.device.type == "cuda")),
            "drop_last": bool(self.args.training.get("drop_last", False)),
            "persistent_workers": bool(self.args.training.get("persistent_workers", workers > 0)) and workers > 0,
            "collate_fn": multi_collate_fn,
        }
        if workers > 0:
            loader_kwargs["prefetch_factor"] = int(self.args.training.get("prefetch_factor", 4))
        return DataLoader(dataset, **loader_kwargs)

    def _to_device(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(self.device, non_blocking=True)
        if isinstance(value, Mapping):
            return {key: self._to_device(item) if key not in {"date_idx", "tick_idxs"} else item for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return type(value)(self._to_device(item) for item in value)
        return value

    def _forward_loss(self, batch):
        output = self.model(batch["feats"])
        prediction, model_loss = self._parse_output(output)
        if model_loss is not None:
            return prediction, model_loss.mean() if model_loss.ndim else model_loss
        if self.loss is None:
            raise ValueError("Self-supervised model must return a loss when args.model.loss.name is 'model'")
        target = self._target_from_batch(batch)
        return prediction, self.loss(prediction, target, {"batch": batch})

    def _parse_output(self, output):
        if isinstance(output, Mapping):
            prediction = output.get(self.prediction_key, output.get(self.representation_key))
            return prediction, output.get("loss")
        if isinstance(output, tuple):
            if len(output) >= 2 and torch.is_tensor(output[1]):
                return output[0], output[1]
            return output[0], None
        return output, None

    def _target_from_batch(self, batch):
        if self.target_key is None:
            raise ValueError("Configured self-supervised loss requires args.model.loss.params.target_key")
        if self.target_key in batch:
            return batch[self.target_key]
        if self.target_key in batch.get("feats", {}):
            return batch["feats"][self.target_key]
        raise KeyError(f"target_key {self.target_key!r} is not present in batch or batch['feats']")

    def _optimizer_step(self):
        if self.args.optimizer.if_grad_norm:
            nn.utils.clip_grad_norm_(self.model.parameters(), float(self.args.optimizer.max_grad_norm))
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _iterate(self, loader, training):
        self.model.train(training)
        values = []
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        for raw_batch in tqdm(loader, desc="Train" if training else "Valid"):
            batch = self._to_device(raw_batch)
            with torch.set_grad_enabled(training):
                _, loss = self._forward_loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite self-supervised loss: {loss.item()}")
                if training:
                    loss.backward()
                    self._optimizer_step()
                values.append(float(loss.detach()))
        return float(np.mean(values)) if values else float("nan")

    def fit(self, train_loader=None, valid_loader=None, save_loss=True, train_range=None, valid_range=None):
        if train_range is None:
            raise ValueError("fit requires explicit train_range")
        train_loader = train_loader or self.make_loader(self.make_dataset(train_range), shuffle=True)
        valid_loader = valid_loader or (self.make_loader(self.make_dataset(valid_range)) if valid_range is not None else None)
        stopper = EarlyStopping(self.args.training.early_stop_patience, self.args.training.early_stop_delta)

        records = []
        for epoch in range(int(self.args.training.num_epoch)):
            train_loss = self._iterate(train_loader, True)
            valid_loss = self._iterate(valid_loader, False) if valid_loader is not None else train_loss
            records.append({"epoch": epoch + 1, "train_loss": train_loss, "valid_loss": valid_loss})
            if self.scheduler is not None:
                self.scheduler.step(valid_loss) if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else self.scheduler.step()
            stopper(valid_loss, self.model, self.model_path)
            if stopper.early_stop:
                break
        frame = pd.DataFrame(records)
        if save_loss:
            frame.to_csv(self.perf_dir / "loss_history.csv", index=False)
        return frame

    @torch.no_grad()
    def predict(self, date_range=None, model_path=None, save=True):
        module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        module.load_state_dict(torch.load(Path(model_path or self.model_path), map_location=self.device))
        module.eval()
        if date_range is None:
            raise ValueError("predict requires explicit date_range")

        dataset = self.make_dataset(date_range)
        loader = self.make_loader(dataset)
        rows = []
        for raw_batch in tqdm(loader, desc="Predict"):
            batch = self._to_device(raw_batch)
            output = module(batch["feats"])
            representation, _ = self._parse_output(output)
            if representation is None:
                continue
            rows.append(
                {
                    "date_idx": np.asarray(raw_batch["date_idx"]).reshape(-1),
                    "tick_idxs": np.asarray(raw_batch["tick_idxs"]).reshape(-1),
                    "values": representation.detach().cpu().numpy(),
                }
            )
        frame = _representation_frame(rows, dataset)
        if save:
            frame.to_csv(self.perf_dir / "representation.csv")
        return frame, None


def _representation_frame(rows, dataset):
    if not rows:
        return pd.DataFrame()
    first_values = rows[0]["values"]
    if first_values.ndim == 1:
        values = np.full((len(dataset.dates), len(dataset.ticks)), np.nan)
        for row in rows:
            values[row["date_idx"][0], row["tick_idxs"]] = row["values"].reshape(-1)
        valid_dates = np.asarray(dataset.dates)[dataset.valid_date_mask]
        return pd.DataFrame(values, index=dataset.dates, columns=dataset.ticks).loc[valid_dates]

    flat_rows = []
    for row in rows:
        date = np.asarray(dataset.dates)[row["date_idx"][0]]
        ticks = np.asarray(dataset.ticks)[row["tick_idxs"]]
        vals = row["values"].reshape(len(ticks), -1)
        for tick, vec in zip(ticks, vals):
            flat_rows.append({"date": date, "tick": tick, **{f"dim_{i}": value for i, value in enumerate(vec)}})
    return pd.DataFrame(flat_rows)
