from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from . import ContextLoss
from .rankic import sigmoid_rank, weighted_corr


class PrimeCombinedLoss(ContextLoss):
    """Cross-sectional PRIME objective with IC, RankIC and hard-pair terms."""

    def __init__(
        self,
        ic_weight: float = 1.0,
        rank_weight: float = 0.5,
        pairwise_weight: float = 0.2,
        physics_weight: float = 0.05,
        vol_neutral_weight: float = 0.05,
        direction_weight: float = 0.01,
        temperature: float = 0.1,
        top_fraction: float = 0.2,
        min_samples: int = 10,
    ):
        super().__init__()
        if not 0 < top_fraction <= 0.5:
            raise ValueError("top_fraction must be in (0, 0.5]")
        self.ic_weight = float(ic_weight)
        self.rank_weight = float(rank_weight)
        self.pairwise_weight = float(pairwise_weight)
        self.physics_weight = float(physics_weight)
        self.vol_neutral_weight = float(vol_neutral_weight)
        self.direction_weight = float(direction_weight)
        self.temperature = float(temperature)
        self.top_fraction = float(top_fraction)
        self.min_samples = int(min_samples)

    def forward(self, preds: Tensor, labels: Tensor, context=None) -> Tensor:
        preds, labels = preds.reshape(-1), labels.reshape(-1)
        valid = torch.isfinite(preds) & torch.isfinite(labels)
        preds, labels = preds[valid], labels[valid]
        if preds.numel() < self.min_samples:
            return preds.sum() * 0.0

        ic_loss = -weighted_corr(preds, labels)
        pred_rank = sigmoid_rank((preds - preds.mean()) / preds.std(unbiased=False).clamp_min(1e-6), self.temperature)
        label_rank = sigmoid_rank((labels - labels.mean()) / labels.std(unbiased=False).clamp_min(1e-6), self.temperature)
        rank_loss = -weighted_corr(pred_rank, label_rank)

        count = max(1, int(preds.numel() * self.top_fraction))
        order = labels.argsort()
        winners, losers = preds[order[-count:]], preds[order[:count]]
        pairwise_loss = F.softplus(-(winners[:, None] - losers[None, :])).mean()

        total = self.ic_weight * ic_loss + self.rank_weight * rank_loss + self.pairwise_weight * pairwise_loss
        output = (context or {}).get("model_output")
        components = output.get("components") if isinstance(output, dict) else None
        if components:
            energy_terms = [components[name].reshape(-1)[valid] for name in ("E_bull", "E_bear", "E_heat")]
            non_negative = sum(F.relu(-term).mean() for term in energy_terms)
            heat_ratio = energy_terms[2].abs() / sum(term.abs() for term in energy_terms).clamp_min(1e-6)
            heat_cap = F.relu(heat_ratio - 0.3).mean()
            total = total + self.physics_weight * (non_negative + heat_cap)
            total = total + self.vol_neutral_weight * weighted_corr(preds, energy_terms[2]).square()

            direction = output.get("direction_regularization")
            if direction is not None:
                total = total + self.direction_weight * direction
        return total
