from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch as th
from torch.utils.data import Dataset

from .datapool import DataPool, ROOT



@dataclass(frozen=True)
class FeatureBlock:
    name: str
    fields: tuple[str, ...]
    lag: int
    kind: str


class BaseDataset(Dataset):
    """Shared DataPool-backed feature reader for batch and flatten datasets."""

    def __init__(
        self,
        dataset_config: Mapping[str, Any],
        feature_blocks: Mapping[str, Any],
        start_date=None,
        end_date=None,
    ):
        self.dataset_config = dict(dataset_config)
        self.feature_blocks = dict(feature_blocks)
        self.root = Path(self.dataset_config.get("root", ROOT))
        self.asset = self.dataset_config.get("asset", "stock")
        self.pool = DataPool(self.root, asset=self.asset)

        self.dates = self.pool.axis.trade_dates.astype("datetime64[D]", copy=False)
        self.ticks = self.pool.axis.ticks
        self.valid_date_mask = np.ones(len(self.dates), dtype=bool)

        start_date = start_date or self.dataset_config.get("start_date")
        end_date = end_date or self.dataset_config.get("end_date")
        if start_date is None or end_date is None:
            raise ValueError("DataPool dataset requires start_date and end_date from the trainer split")
        self.start_idx = self.pool.axis.date_position(start_date)
        self.end_idx = self.pool.axis.date_position(end_date)
        self.mode = self.dataset_config.get("mode", "universe")
        self.pool_name = self.dataset_config.get("pool_name")
        self.fix_stock = self.dataset_config.get("fix_stock")
        self.sample_size = self.dataset_config.get("sample_size")
        self.nan_filter_blocks = set(self.dataset_config.get("nanflit_set") or self.dataset_config.get("nanfilt_set") or ["dailyset"])

        self.label_name = self._normalize_label(self.dataset_config.get("label"))
        self.label_array = self.pool.load(self.label_name) if self.label_name else None
        self.tradable = self._optional_load(self.pool, "mask/tradable")
        self.pool_mask = self._optional_load(self.pool, f"mask/{self.pool_name}_mask") if self.mode == "pool" else None
        self.fix_tick_indices = self.pool.axis.tick_positions(self.fix_stock) if self.mode == "fix" else None

        self.blocks = self._build_blocks(self.feature_blocks)
        self.field_arrays = {
            field: self.pool.load(field)
            for block in self.blocks.values()
            for field in block.fields
        }
        self.max_lag = max((block.lag for block in self.blocks.values() if block.kind == "daily"), default=1)
        self.date_indices = np.arange(max(self.start_idx, self.max_lag - 1), self.end_idx + 1, dtype=np.int64)
        self._candidate_ticks = [self._select_candidate_ticks(date_idx) for date_idx in self.date_indices]

    @staticmethod
    def _normalize_label(label):
        if not label:
            return None
        name = str(label).strip().removesuffix(".bin")
        if "/" in name:
            return name
        return f"model_input/labels/{name}"

    @staticmethod
    def _optional_load(pool: DataPool, field):
        try:
            return pool.load(field)
        except FileNotFoundError:
            return None

    @staticmethod
    def _block_kind(name: str, cfg: Mapping[str, Any]) -> str:
        if "kind" in cfg:
            return str(cfg["kind"]).lower()
        if "freq" in cfg:
            return "minute" if str(cfg["freq"]).lower() in {"minute", "intraday"} else "daily"
        return "minute" if name == "minuteset" or str(name).startswith("minuteset") else "daily"

    @classmethod
    def _build_blocks(cls, specified: Mapping[str, Any]) -> dict[str, FeatureBlock]:
        blocks = {}
        for name, cfg in specified.items():
            kind = cls._block_kind(str(name), cfg)
            if kind not in {"daily", "minute"}:
                continue
            default_path = "model_input/dGRU" if kind == "daily" else "m_essentials"
            fields = tuple(str(field).strip().removesuffix(".bin") for field in cfg.get("fields", ()))
            data_path = str(cfg.get("data_path", default_path)).replace(chr(92), "/").strip("/")
            normalized = tuple(field if "/" in field else f"{data_path}/{field}" for field in fields)
            blocks[str(name)] = FeatureBlock(
                name=str(name),
                fields=normalized,
                lag=int(cfg.get("lag", 1) or 1),
                kind=kind,
            )
        if not blocks:
            raise ValueError("DataPool dataset needs at least one daily or minute feature block")
        return blocks

    def _select_candidate_ticks(self, date_idx: int) -> np.ndarray:
        if self.tradable is None:
            valid = np.ones(len(self.ticks), dtype=bool)
        else:
            valid = np.asarray(self.tradable[date_idx, :len(self.ticks)], dtype=bool).copy()
        if self.label_array is not None:
            valid &= np.isfinite(self.label_array[date_idx, :len(self.ticks)])
        if self.mode == "pool":
            if self.pool_mask is None:
                raise FileNotFoundError(f"pool mask is missing: {self.pool_name}")
            valid &= np.asarray(self.pool_mask[date_idx, :len(self.ticks)], dtype=bool)
            ticks = np.flatnonzero(valid)
        elif self.mode == "fix":
            ticks = self.fix_tick_indices
            ticks = ticks[valid[ticks]]
        elif self.mode == "sample":
            ticks = np.flatnonzero(valid)
            if self.sample_size and len(ticks) > int(self.sample_size):
                rng = np.random.default_rng(date_idx)
                ticks = np.sort(rng.choice(ticks, int(self.sample_size), replace=False))
        elif self.mode == "universe":
            ticks = np.flatnonzero(valid)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        return ticks.astype(np.int64, copy=False)

    def _read_field(self, field: str, block: FeatureBlock, date_idx: int, ticks: np.ndarray) -> np.ndarray:
        arr = self.field_arrays[field]
        if block.kind == "daily":
            start = date_idx - block.lag + 1
            return np.asarray(arr[start:date_idx + 1, ticks], dtype=np.float32).T
        if arr.ndim != 3:
            raise ValueError(f"minute block {block.name} expects 3-D field, got {field}: {arr.shape}")
        return np.asarray(arr[date_idx, :, ticks], dtype=np.float32).T

    def _read_block(self, block: FeatureBlock, date_idx: int, ticks: np.ndarray) -> np.ndarray:
        arrays = [self._read_field(field, block, date_idx, ticks) for field in block.fields]
        return np.nan_to_num(np.stack(arrays, axis=-1), nan=0.0, posinf=0.0, neginf=0.0)

    def _read_features(self, date_idx: int, ticks: np.ndarray) -> dict[str, np.ndarray]:
        return {name: self._read_block(block, date_idx, ticks) for name, block in self.blocks.items()}

    def _filter_ticks(self, ticks: np.ndarray, feats: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if not self.nan_filter_blocks:
            return ticks, feats
        keep = np.ones(len(ticks), dtype=bool)
        for name in self.nan_filter_blocks:
            if name in feats:
                keep &= ~np.all(np.isnan(feats[name]), axis=-1).any(axis=1)
        return ticks[keep], {name: value[keep] for name, value in feats.items()}

    def _label(self, date_idx: int, ticks: np.ndarray) -> np.ndarray:
        if self.label_array is None:
            return np.full(len(ticks), np.nan, dtype=np.float32)
        return np.asarray(self.label_array[date_idx, ticks], dtype=np.float32)

    @staticmethod
    def _tensorize(feats: dict[str, np.ndarray]) -> dict[str, th.Tensor]:
        return {name: th.from_numpy(value).float().contiguous() for name, value in feats.items()}

    def close(self):
        self.pool.close()


class BatchDataset(BaseDataset):
    """One item is one trading date cross-section."""

    def __init__(self, dataset_config: Mapping[str, Any], feature_blocks: Mapping[str, Any], **kwargs):
        super().__init__(dataset_config=dataset_config, feature_blocks=feature_blocks, **kwargs)
        self.data_map = {idx: int(date_idx) for idx, date_idx in enumerate(self.date_indices)}

    def __len__(self):
        return len(self.date_indices)

    def __getitem__(self, idx):
        date_idx = self.data_map[int(idx)]
        ticks = self._candidate_ticks[int(idx)]
        feats = self._read_features(date_idx, ticks)
        ticks, feats = self._filter_ticks(ticks, feats)
        return {
            "feats": self._tensorize(feats),
            "label": th.from_numpy(self._label(date_idx, ticks)).contiguous(),
            "date_idx": date_idx,
            "tick_idxs": ticks,
        }


class FlattenDataset(BaseDataset):
    """One item is one (date, stock) sample."""

    def __init__(self, dataset_config: Mapping[str, Any], feature_blocks: Mapping[str, Any], **kwargs):
        super().__init__(dataset_config=dataset_config, feature_blocks=feature_blocks, **kwargs)
        pairs = []
        for local_date_idx, date_idx in enumerate(self.date_indices):
            for tick in self._candidate_ticks[local_date_idx]:
                pairs.append((int(date_idx), int(tick)))
        self.data_map = pairs

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        date_idx, tick = self.data_map[int(idx)]
        ticks = np.asarray([tick], dtype=np.int64)
        feats = self._read_features(date_idx, ticks)
        label = self._label(date_idx, ticks)
        return {
            "feats": {name: th.from_numpy(value.squeeze(0)).float().contiguous() for name, value in feats.items()},
            "label": th.from_numpy(label).float().contiguous(),
            "date_idx": date_idx,
            "tick_idxs": tick,
        }


DataPoolDailyBatchDataset = DataPoolBatchDataset





