'''
DomainProvider (ABC)                      ← 抽象基类，定义接口
   ├── _AlignedBinaryProvider             ← 公共底座：加载日期/股票轴，做索引对齐
   │      ├── IndexDomainProvider         ← 按指数成分生成 mask
   │      └── IndustryDomainProvider      ← 按行业代码生成 mask
   └── (未来可扩展：概念、相似度等)

DOMAIN_PROVIDERS / register_domain_provider / build_domain_provider  ← 注册表 + 工厂，按名字创建 Provider
'''


from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from v3.paths import DATA_ROOT
from .rankic import DifferentiableRankICLoss

ROOT = DATA_ROOT


class DomainProvider(ABC):
    """Map a date and ticker cross-section to named boolean domains."""

    @abstractmethod
    def get(self, date: str, ticks: Sequence[str]) -> Mapping[str, np.ndarray]:
        """Return {domain_name: membership_mask}, aligned with ticks."""


DOMAIN_PROVIDERS = {}


def register_domain_provider(name: str, provider_class) -> None:
    """Register a provider class, including future concept/similarity providers."""
    key = name.lower()
    if not key:
        raise ValueError("domain provider name must not be empty")
    DOMAIN_PROVIDERS[key] = provider_class


def build_domain_provider(name: str, params=None) -> DomainProvider:
    try:
        provider_class = DOMAIN_PROVIDERS[name.lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown domain provider {name!r}; available: {sorted(DOMAIN_PROVIDERS)}") from exc
    return provider_class(**dict(params or {}))


class _AlignedBinaryProvider(DomainProvider):
    def __init__(self, axis_root=ROOT/"axis", ticks_file="ticks.npy"):
        self.axis_root = Path(axis_root)
        self.dates = np.load(self.axis_root / "dates.npy", allow_pickle=True).astype(str)
        self.ticks = np.load(self.axis_root / ticks_file, allow_pickle=True).astype(str)
        self.shape = (len(self.dates), len(self.ticks))
        self.date_lookup = {value: index for index, value in enumerate(self.dates) if value != "NaT"}
        self.tick_lookup = {value: index for index, value in enumerate(self.ticks)}

    def _indices(self, date: str, ticks: Sequence[str]):
        date_index = self.date_lookup.get(str(date))
        if date_index is None:
            raise KeyError(f"{date} is missing from {self.axis_root}")
        tick_indices = np.fromiter((self.tick_lookup.get(str(tick), -1) for tick in ticks), dtype=np.int64)
        return date_index, tick_indices, tick_indices >= 0


class IndexDomainProvider(_AlignedBinaryProvider):
    """Build index-universe domains from aligned constituent mask files."""

    def __init__(self, axis_root=ROOT/"axis", mask_root=ROOT/"mask",
                 domains=("hs300", "zz500", "zz1000", "others"), ticks_file="ticks.npy"):
        super().__init__(axis_root, ticks_file)
        self.mask_root = Path(mask_root)
        self.domains = tuple(domains)
        self.masks = {
            name: np.memmap(self.mask_root / f"{name}_mask.bin", dtype=bool, mode="r", shape=self.shape)
            for name in ("hs300", "zz500", "zz1000")
        }

    def get(self, date: str, ticks: Sequence[str]) -> Mapping[str, np.ndarray]:
        date_index, tick_indices, known = self._indices(date, ticks)
        masks = {}
        for name in ("hs300", "zz500", "zz1000"):
            values = np.zeros(len(ticks), dtype=bool)
            values[known] = self.masks[name][date_index, tick_indices[known]]
            masks[name] = values
        masks["zz800"] = masks["hs300"] | masks["zz500"]
        masks["others"] = known & ~(masks["zz800"] | masks["zz1000"])
        try:
            return {name: masks[name] for name in self.domains}
        except KeyError as exc:
            raise ValueError(f"Unsupported index domain {exc.args[0]!r}") from exc


class IndustryDomainProvider(_AlignedBinaryProvider):
    """Build one domain per industry code from an aligned numeric mask file."""

    def __init__(self, axis_root=ROOT/"axis", mask_root=ROOT/"mask", domains=None,
                 mask_file="industry.bin", dtype="float64", ticks_file="ticks.npy"):
        super().__init__(axis_root, ticks_file)
        self.mask_root = Path(mask_root)
        self.domains = None if domains is None else tuple(domains)
        self.industry = np.memmap(
            self.mask_root / mask_file, dtype=np.dtype(dtype), mode="r", shape=self.shape
        )

    def get(self, date: str, ticks: Sequence[str]) -> Mapping[str, np.ndarray]:
        date_index, tick_indices, known = self._indices(date, ticks)
        codes = np.full(len(ticks), np.nan, dtype=float)
        codes[known] = self.industry[date_index, tick_indices[known]]
        requested = self.domains
        if requested is None:
            requested = tuple(str(value) for value in np.unique(codes[np.isfinite(codes)]))
        return {str(code): known & np.isfinite(codes) & (codes == float(code)) for code in requested}


register_domain_provider("index", IndexDomainProvider)
register_domain_provider("industry", IndustryDomainProvider)


class DomainWeightedRankICLoss(DifferentiableRankICLoss):
    """Weighted RankIC across domains supplied by a pluggable provider."""

    def __init__(self, temperature: float = 0.01, method: str = "sigmoid",
                 domain_type: str = "index", domains=None, domain_weights=None,
                 provider_params=None, axis_root=None, mask_root=None, min_samples: int = 2):
        super().__init__(temperature, method)
        params = dict(provider_params or {})
        if domains is not None:
            params["domains"] = domains
        if axis_root is not None:
            params["axis_root"] = axis_root
        if mask_root is not None:
            params["mask_root"] = mask_root
        self.provider = build_domain_provider(domain_type, params)
        self.domain_weights = None if domain_weights is None else tuple(float(v) for v in domain_weights)
        if self.domain_weights is not None and domains is None:
            raise ValueError("domains must be explicit when domain_weights are supplied")
        if self.domain_weights is not None and len(domains) != len(self.domain_weights):
            raise ValueError("domains and domain_weights must have equal length")
        self.min_samples = int(min_samples)
        self.dataset = None

    def configure_dataset(self, dataset) -> None:
        self.dataset = getattr(dataset, "dataset", dataset)

    def forward(self, preds: Tensor, labels: Tensor, context=None) -> Tensor:
        if context is None or self.dataset is None:
            raise RuntimeError("Domain RankIC needs a configured dataset and batch context")
        date_ids = np.asarray(context["date_idx"]).reshape(-1)
        if len(date_ids) != 1:
            raise ValueError("Domain RankIC requires batch_size=1: one daily cross-section")
        tick_ids = np.asarray(context["tick_idxs"]).reshape(-1)
        memberships = self.provider.get(
            np.asarray(self.dataset.dates)[date_ids[0]], np.asarray(self.dataset.ticks)[tick_ids]
        )
        valid_domains = [
            (name, mask) for name, mask in memberships.items()
            if int(np.asarray(mask, dtype=bool).sum()) >= self.min_samples
        ]
        if not valid_domains:
            return preds.sum() * 0.0
        if self.domain_weights is None:
            coefficients = [1.0 / len(valid_domains)] * len(valid_domains)
        else:
            configured = dict(zip(memberships, self.domain_weights))
            coefficients = [configured[name] for name, _ in valid_domains]
        loss = preds.sum() * 0.0
        for coefficient, (_, membership) in zip(coefficients, valid_domains):
            mask = torch.as_tensor(membership, device=preds.device, dtype=torch.bool)
            if coefficient:
                loss = loss - coefficient * self.correlation(
                    preds.reshape(-1)[mask], labels.reshape(-1)[mask]
                )
        return loss
