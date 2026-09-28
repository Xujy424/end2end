from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F

from . import ContextLoss


class PairwiseRankLoss(ContextLoss):
    """RankNet-style loss on high-return winners and low-return losers."""

    def __init__(
        self,
        top_fraction: float = 0.2,
        temperature: float = 1.0,
        margin: float = 0.0,
        min_label_gap: float = 0.0,
        max_pairs: int = 65536,
        min_samples: int = 10,
    ):
        super().__init__()
        if not 0 < top_fraction <= 0.5:
            raise ValueError("top_fraction must be in (0, 0.5]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if max_pairs <= 0:
            raise ValueError("max_pairs must be positive")
        self.top_fraction = float(top_fraction)
        self.temperature = float(temperature)
        self.margin = float(margin)
        self.min_label_gap = float(min_label_gap)
        self.max_pairs = int(max_pairs)
        self.min_samples = int(min_samples)

    @staticmethod
    def finite_inputs(preds: Tensor, labels: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        preds, labels = preds.reshape(-1), labels.reshape(-1)
        valid = torch.isfinite(preds) & torch.isfinite(labels)
        return preds[valid], labels[valid], valid

    def pairwise_loss(self, preds: Tensor, labels: Tensor) -> Tensor:
        if preds.numel() < self.min_samples:
            return preds.sum() * 0.0

        count = max(1, int(preds.numel() * self.top_fraction))
        count = min(count, max(1, math.isqrt(self.max_pairs)))
        order = labels.argsort()
        winner_indices = order[-count:]
        loser_indices = order[:count]

        score_gap = preds[winner_indices, None] - preds[loser_indices][None, :]
        label_gap = labels[winner_indices, None] - labels[loser_indices][None, :]
        valid_pairs = label_gap > self.min_label_gap
        if not torch.any(valid_pairs):
            return preds.sum() * 0.0

        logits = (score_gap[valid_pairs] - self.margin) / self.temperature
        return F.softplus(-logits).mean()

    def forward(self, preds: Tensor, labels: Tensor, context=None) -> Tensor:
        preds, labels, _ = self.finite_inputs(preds, labels)
        return self.pairwise_loss(preds, labels)
