from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .rankic import DifferentiableRankICLoss


class IndexDomainProvider:
    """Read Z-drive masks and align them by date/ticker, never by raw offsets."""

    def __init__(self, axis_root="Z:/axis", mask_root="Z:/stock/index/mask", domains=("zz800", "zz1000", "others")):
        self.axis_root = Path(axis_root)
        self.mask_root = Path(mask_root)
        self.domains = tuple(domains)
        self.dates = np.load(self.axis_root / "dates.npy", allow_pickle=False).astype(str)
        self.ticks = np.load(self.axis_root / "stock_ticks.npy", allow_pickle=False).astype(str)
        self.shape = (len(self.dates), len(self.ticks))
        self.date_lookup = {value: index for index, value in enumerate(self.dates) if value != "NaT"}
        self.tick_lookup = {value: index for index, value in enumerate(self.ticks)}
        self.masks = {
            name: np.memmap(self.mask_root / f"{name}_mask.bin", dtype=bool, mode="r", shape=self.shape)
            for name in ("hs300", "zz500", "zz1000")
        }

    def get(self, date: str, ticks: Sequence[str]) -> np.ndarray:
        date_index = self.date_lookup.get(str(date))
        if date_index is None:
            raise KeyError(f"{date} is missing from {self.axis_root}")
        tick_indices = np.fromiter((self.tick_lookup.get(str(tick), -1) for tick in ticks), dtype=np.int64)
        known = tick_indices >= 0
        masks = {}
        for name in ("hs300", "zz500", "zz1000"):
            values = np.zeros(len(ticks), dtype=bool)
            values[known] = self.masks[name][date_index, tick_indices[known]]
            masks[name] = values
        masks["zz800"] = masks["hs300"] | masks["zz500"]
        masks["others"] = known & ~(masks["zz800"] | masks["zz1000"])
        try:
            return np.stack([masks[name] for name in self.domains], axis=1)
        except KeyError as exc:
            raise ValueError(f"Unsupported domain {exc.args[0]!r}") from exc


class DomainWeightedRankICLoss(DifferentiableRankICLoss):
    def __init__(
        self,
        temperature: float = 0.01,
        method: str = "sigmoid",
        domains=("zz800", "zz1000", "others"),
        domain_weights=(0.025, 0.8, 0.175),
        axis_root="Z:/axis",
        mask_root="Z:/stock/index/mask",
        min_samples: int = 2,
    ):
        super().__init__(temperature, method)
        if len(domains) != len(domain_weights):
            raise ValueError("domains and domain_weights must have equal length")
        self.domain_weights = tuple(float(value) for value in domain_weights)
        self.min_samples = int(min_samples)
        self.provider = IndexDomainProvider(axis_root, mask_root, domains)
        self.dataset = None

    def configure_dataset(self, dataset) -> None:
        self.dataset = getattr(dataset, "dataset", dataset)

    def forward(self, preds: Tensor, labels: Tensor, context=None) -> Tensor:
        if context is None or self.dataset is None:
            raise RuntimeError("Domain RankIC needs a configured dataset and batch context")
        date_ids = np.asarray(context["date_idx"]).reshape(-1)
        if len(date_ids) != 1:
            raise ValueError("Domain RankIC requires batch_size=1: one daily N×T×F cross-section")
        tick_ids = np.asarray(context["tick_idxs"]).reshape(-1)
        memberships = self.provider.get(
            np.asarray(self.dataset.dates)[date_ids[0]],
            np.asarray(self.dataset.ticks)[tick_ids],
        )
        loss = preds.sum() * 0.0
        used = False
        for index, coefficient in enumerate(self.domain_weights):
            mask = torch.as_tensor(memberships[:, index], device=preds.device, dtype=torch.bool)
            if coefficient and int(mask.sum()) >= self.min_samples:
                loss = loss - coefficient * self.correlation(preds.reshape(-1)[mask], labels.reshape(-1)[mask])
                used = True
        return loss if used else preds.sum() * 0.0
