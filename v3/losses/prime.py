from __future__ import annotations

from torch import Tensor
import torch.nn.functional as F

from .pairwise import PairwiseRankLoss
from .rankic import sigmoid_rank, weighted_corr


class PrimeCombinedLoss(PairwiseRankLoss):
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
        pairwise_temperature: float = 1.0,
        top_fraction: float = 0.2,
        pairwise_margin: float = 0.0,
        min_label_gap: float = 0.0,
        max_pairs: int = 65536,
        min_samples: int = 10,
    ):
        super().__init__(
            top_fraction=top_fraction,
            temperature=pairwise_temperature,
            margin=pairwise_margin,
            min_label_gap=min_label_gap,
            max_pairs=max_pairs,
            min_samples=min_samples,
        )
        self.ic_weight = float(ic_weight)
        self.rank_weight = float(rank_weight)
        self.pairwise_weight = float(pairwise_weight)
        self.physics_weight = float(physics_weight)
        self.vol_neutral_weight = float(vol_neutral_weight)
        self.direction_weight = float(direction_weight)
        self.rank_temperature = float(temperature)

    def forward(self, preds: Tensor, labels: Tensor, context=None) -> Tensor:
        preds, labels, valid = self.finite_inputs(preds, labels)
        if preds.numel() < self.min_samples:
            return preds.sum() * 0.0

        ic_loss = -weighted_corr(preds, labels)
        pred_rank = sigmoid_rank((preds - preds.mean()) / preds.std(unbiased=False).clamp_min(1e-6), self.rank_temperature)
        label_rank = sigmoid_rank((labels - labels.mean()) / labels.std(unbiased=False).clamp_min(1e-6), self.rank_temperature)
        rank_loss = -weighted_corr(pred_rank, label_rank)
        pairwise_loss = self.pairwise_loss(preds, labels)

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
