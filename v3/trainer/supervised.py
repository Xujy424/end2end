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

from v3.dataset import DATASET_DICT
from v3.losses import build_loss
from v3.optimizers import EarlyStopping, build_optimizer_bundle


class SupervisedTrainerV3:
    """Context-aware daily supervised trainer."""

    def __init__(self, args, model_class, loss=None, run_name=None):
        self.args = copy.deepcopy(args)
        self.device = torch.device(self.args.training.device if torch.cuda.is_available() else "cpu")
        self._set_seed(int(self.args.training.seed))

        self.model = model_class(**self.args.model.params).to(self.device).float()
        if torch.cuda.device_count() > 1 and self.args.training.multi_gpu:
            self.model = nn.DataParallel(
                self.model, 
                device_ids=list(self.args.training.available_gpu), 
                output_device=int(self.args.training.main_gpu)
            )

        self._prepare_model()

        self.loss = loss or build_loss(
            self.args.loss.name, 
            self.args.loss.get("params", {})
        )
        self.optimizer, self.scheduler = build_optimizer_bundle(
            self.args.optimizer,
            (p for p in self.model.parameters() if p.requires_grad),
        )

        suffix = run_name or self.args.loss.name
        self.perf_dir = Path(self.args.training.perf_path).expanduser() / self.args.model.name / "v3" / suffix
        self.perf_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.perf_dir / "best_model.pth"

    def _prepare_model(self):
        """Hook for trainers that load or freeze model parameters before optimization."""

    def _set_model_mode(self, training):
        self.model.train(training)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def make_dataset(self, date_range):
        params = copy.deepcopy(self.args.dataset.params)
        return DATASET_DICT[self.args.dataset.name](
            start_date=date_range[0],
            end_date=date_range[1],
            **params,
        )

    def make_loader(self, dataset, *, shuffle=False, indices=None):
        if getattr(self.loss, "requires_ordered_batches", False) and shuffle:
            raise ValueError("Temporal RankIC requires chronological batches; shuffle must be False")
        
        loader_dataset = Subset(dataset, list(indices)) if indices is not None else dataset

        workers = int(self.args.training.get("num_workers", 0))
        loader_kwargs = {
            "batch_size": None,
            "shuffle": shuffle,
            "num_workers": workers,
            "pin_memory": bool(self.args.training.get("pin_memory", self.device.type == "cuda")),
            "persistent_workers": bool(self.args.training.get("persistent_workers", workers > 0)) and workers > 0,
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
        self._set_model_mode(training)
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
                temporal = self._temporal_context(raw_batch, history[0]) if ordered and len(history) == lag else None   # 共同股票位置：共同股票预测值
                snapshot = {int(tick): pred.detach().cpu() for tick, pred in zip(np.asarray(raw_batch["tick_idxs"]).reshape(-1), preds)}  # 当前股票idx：当前预测值
                history.append(snapshot)
                if ordered and temporal is None:
                    continue
                if temporal:
                    temporal = {key: item.to(self.device) for key, item in temporal.items()}  # 搬到GPU
                    context.update(temporal)   # date_idx, tick_idxs, current_indices, previous_preds

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

    
    def fit(
        self,
        train_loader=None,
        valid_loader=None,
        save_loss=True,
        train_range=None,
        valid_range=None,
        warm_start_path=None,
    ):
        ordered = getattr(self.loss, "requires_ordered_batches", False)

        if train_range is None or valid_range is None:
            raise ValueError("fit requires explicit train_range and valid_range")
        
        train_loader = train_loader or self.make_loader(self.make_dataset(train_range), shuffle=not ordered)
        valid_loader = valid_loader or self.make_loader(self.make_dataset(valid_range))

        records = []
        initial_best_loss = np.inf
        if warm_start_path is not None:
            self._load_model(warm_start_path)
            initial_best_loss = self._iterate(valid_loader, False)
            if not np.isfinite(initial_best_loss):
                raise FloatingPointError(f"Non-finite warm-start validation loss: {initial_best_loss}")
            self._save_model(self.model_path)
            records.append({"epoch": 0, "train_loss": np.nan, "valid_loss": initial_best_loss})
            print(f"Warm start | valid_loss={initial_best_loss:.6f}")

        stopper = EarlyStopping(
            self.args.training.early_stop_patience,
            self.args.training.early_stop_delta,
            best_loss=initial_best_loss,
        )

        for epoch in range(int(self.args.training.num_epoch)):
            train_loss = self._iterate(train_loader, True)
            valid_loss = self._iterate(valid_loader, False)
            records.append({"epoch": epoch + 1, "train_loss": train_loss, "valid_loss": valid_loss})
            print(f"Epoch {epoch + 1:03d} | train_loss={train_loss:.6f} | valid_loss={valid_loss:.6f}")
            if self.scheduler is not None:
                self.scheduler.step(valid_loss) if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else self.scheduler.step()
            stopper(valid_loss, self.model, self.model_path)
            if stopper.early_stop:
                break
        frame = pd.DataFrame(records)
        if save_loss:
            frame.to_csv(self.perf_dir / "loss_history.csv", index=False)
        return frame

    def _load_model(self, model_path):
        module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        module.load_state_dict(torch.load(Path(model_path), map_location=self.device))

    def _save_model(self, model_path):
        module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(module.state_dict(), Path(model_path))


    @torch.no_grad()
    def predict(self, date_range=None, model_path=None, save=True):
        self._load_model(model_path or self.model_path)
        module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        module.eval()

        if date_range is None:
            raise ValueError("predict requires explicit date_range")
        dataset = self.make_dataset(date_range)
        loader = self.make_loader(dataset)
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

        valid_dates = np.asarray(dataset.dates)[dataset.valid_date_mask]  # 这里的validmask是不是不太需要
        pred_df = pd.DataFrame(preds, index=dataset.dates, columns=dataset.ticks).loc[valid_dates]
        label_df = pd.DataFrame(labels, index=dataset.dates, columns=dataset.ticks).loc[valid_dates]
        if save:
            pred_df.to_csv(self.perf_dir / "alpha.csv")
            label_df.to_csv(self.perf_dir / "label.csv")
        return pred_df, label_df



