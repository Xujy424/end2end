from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from v3.dataset import DATASET_DICT, multi_collate_fn
from v3.training.losses import build_loss
from v3.training.optimizers import EarlyStopping, build_optimizer, build_scheduler


class SupervisedTrainerV3:
    """Context-aware daily trainer kept separate from the legacy framework."""

    def __init__(self, args, model_class, loss=None, run_name=None):
        self.args = copy.deepcopy(args)
        self.device = torch.device(self.args.training.device if torch.cuda.is_available() else "cpu")
        self._set_seed(int(self.args.training.seed))
        self.model = model_class(**self.args.model.params).to(self.device).float()
        if torch.cuda.device_count() > 1 and self.args.training.multi_gpu:
            self.model = nn.DataParallel(self.model, device_ids=list(self.args.training.available_gpu), output_device=int(self.args.training.main_gpu))
        self.loss = loss or build_loss(self.args.model.loss.name, self.args.model.loss.get("params", {}))
        self.optimizer = build_optimizer(self.args.optimizer.name, (p for p in self.model.parameters() if p.requires_grad), self.args.optimizer.optim_params)
        self.scheduler = build_scheduler(self.args.optimizer.scheduler, self.optimizer, self.args.optimizer.sched_params) if self.args.optimizer.if_lr_decay else None
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

    def _dataset(self, start_date, end_date):
        params = copy.deepcopy(self.args.training.dataset.params)
        return DATASET_DICT[self.args.training.dataset.name](start_date=start_date, end_date=end_date, **params)

    def make_loader(self, start_date, end_date, *, shuffle=False, indices=None):
        if getattr(self.loss, "requires_ordered_batches", False) and shuffle:
            raise ValueError("Temporal RankIC requires chronological batches; shuffle must be False")
        if int(self.args.training.batch_size) != 1:
            raise ValueError("V3 requires batch_size=1: one daily N闂佺厧顕悥锕傛煠鐎靛摜鍙?cross-section")
        dataset = self._dataset(start_date, end_date)
        loader_dataset = Subset(dataset, list(indices)) if indices is not None else dataset
        workers = int(self.args.training.get("num_workers", 0))
        loader_kwargs = {
            "batch_size": 1,
            "shuffle": shuffle,
            "num_workers": workers,
            "pin_memory": bool(self.args.training.get("pin_memory", self.device.type == "cuda")),
            "drop_last": False,
            "persistent_workers": bool(self.args.training.get("persistent_workers", workers > 0)) and workers > 0,
            "collate_fn": multi_collate_fn,
        }
        if workers > 0:
            loader_kwargs["prefetch_factor"] = int(self.args.training.get("prefetch_factor", 4))
        return DataLoader(loader_dataset, **loader_kwargs)

    def _to_device(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(self.device, non_blocking=True)
        if isinstance(value, Mapping):
            return {key: self._to_device(item) if key not in {"date_idx", "tick_idxs"} else item for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return type(value)(self._to_device(item) for item in value)
        return value

    @staticmethod
    def _temporal_context(raw_batch, previous):
        ticks = np.asarray(raw_batch["tick_idxs"]).reshape(-1)
        current_lookup = {int(tick): index for index, tick in enumerate(ticks)}
        common = [tick for tick in previous if tick in current_lookup]
        if len(common) < 2:
            return None
        return {
            "current_indices": torch.as_tensor([current_lookup[tick] for tick in common], dtype=torch.long),
            "previous_preds": torch.stack([previous[tick] for tick in common]),
        }

    def _optimizer_step(self):
        if self.args.optimizer.if_grad_norm:
            nn.utils.clip_grad_norm_(self.model.parameters(), float(self.args.optimizer.max_grad_norm))
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _iterate(self, loader, training):
        self.loss.configure_dataset(loader.dataset)
        self.model.train(training)
        ordered = getattr(self.loss, "requires_ordered_batches", False)
        lag = int(self.args.training.get("prediction_lag", 1))
        history = deque(maxlen=lag)
        values = []
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        denominator = max(len(loader) - lag, 1) if ordered else max(len(loader), 1)
        for raw_batch in tqdm(loader, desc="Train" if training else "Valid"):
            batch = self._to_device(raw_batch)
            with torch.set_grad_enabled(training):
                preds = self.model(batch["feats"]).reshape(-1)
                context = {"date_idx": raw_batch["date_idx"], "tick_idxs": raw_batch["tick_idxs"]}
                temporal = self._temporal_context(raw_batch, history[0]) if ordered and len(history) == lag else None
                snapshot = {int(tick): pred.detach().cpu() for tick, pred in zip(np.asarray(raw_batch["tick_idxs"]).reshape(-1), preds)}
                history.append(snapshot)
                if ordered and temporal is None:
                    continue
                if temporal:
                    temporal = {key: item.to(self.device) for key, item in temporal.items()}
                    context.update(temporal)
                loss = self.loss(preds, batch["label"].reshape(-1), context)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite {type(self.loss).__name__}: {loss.item()}")
                if training:
                    scaled_loss = loss / denominator if self.loss.requires_epoch_update else loss
                    scaled_loss.backward()
                    if not getattr(self.loss, "requires_epoch_update", False):
                        self._optimizer_step()
                values.append(float(loss.detach()))
        if training and getattr(self.loss, "requires_epoch_update", False) and values:
            self._optimizer_step()
        return float(np.mean(values)) if values else float("nan")
    def fit(self, train_loader=None, valid_loader=None, save_loss=True, train_range=None, valid_range=None):
        ordered = getattr(self.loss, "requires_ordered_batches", False)
        if train_range is None or valid_range is None:
            raise ValueError("fit requires explicit train_range and valid_range")
        train_loader = train_loader or self.make_loader(*train_range, shuffle=not ordered)
        valid_loader = valid_loader or self.make_loader(*valid_range)
        stopper = EarlyStopping(self.args.training.early_stop_patience, self.args.training.early_stop_delta)
        records = []
        for epoch in range(int(self.args.training.num_epoch)):
            train_loss = self._iterate(train_loader, True)
            valid_loss = self._iterate(valid_loader, False)
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
        loader = self.make_loader(*date_range)
        dataset = loader.dataset
        preds = np.full((len(dataset.dates), len(dataset.ticks)), np.nan)
        labels = np.full_like(preds, np.nan)
        for raw_batch in tqdm(loader, desc="Predict"):
            batch = self._to_device(raw_batch)
            output = module(batch["feats"]).reshape(-1).cpu().numpy()
            date_ids = np.asarray(raw_batch["date_idx"]).reshape(-1)
            ticks = np.asarray(raw_batch["tick_idxs"]).reshape(-1)
            preds[date_ids[0], ticks] = output
            labels[date_ids[0], ticks] = batch["label"].cpu().numpy().reshape(-1)
        valid_dates = np.asarray(dataset.dates)[dataset.valid_date_mask]
        pred_df = pd.DataFrame(preds, index=dataset.dates, columns=dataset.ticks).loc[valid_dates]
        label_df = pd.DataFrame(labels, index=dataset.dates, columns=dataset.ticks).loc[valid_dates]
        if save:
            pred_df.to_csv(self.perf_dir / "alpha.csv")
            label_df.to_csv(self.perf_dir / "label.csv")
        return pred_df, label_df





