from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from . import ContextLoss


def finite_vectors(*values: Tensor) -> tuple[Tensor, ...]:
    values = tuple(value.reshape(-1) for value in values)
    valid = torch.ones_like(values[0], dtype=torch.bool)
    for value in values:
        valid &= torch.isfinite(value)
    return tuple(value[valid] for value in values)


def weighted_corr(x: Tensor, y: Tensor, weight: Optional[Tensor] = None, eps: float = 1e-8) -> Tensor:
    x, y = finite_vectors(x, y)
    if x.numel() < 2:
        return x.sum() * 0.0
    if weight is None:
        weight = torch.ones_like(x)
    else:
        weight = weight.reshape(-1).to(device=x.device, dtype=x.dtype)
        valid = torch.isfinite(weight) & (weight > 0)
        x, y, weight = x[valid], y[valid], weight[valid]
    if x.numel() < 2 or weight.sum() <= eps:
        return x.sum() * 0.0
    weight = weight / weight.sum()
    x = x - (weight * x).sum()
    y = y - (weight * y).sum()
    denominator = (weight * x.square()).sum().sqrt() * (weight * y.square()).sum().sqrt()
    return (weight * x * y).sum() / denominator.clamp_min(eps)


def sigmoid_rank(values: Tensor, temperature: float = 0.1) -> Tensor:
    values = values.reshape(-1)
    differences = (values.unsqueeze(0) - values.unsqueeze(1)) / temperature
    return torch.sigmoid(differences).sum(dim=1) - 0.5


def neural_sort_rank(values: Tensor, temperature: float = 0.1) -> Tensor:
    values = values.reshape(-1, 1)
    n = values.shape[0]
    absolute_differences = torch.abs(values - values.T)
    scaling = n + 1 - 2 * torch.arange(1, n + 1, device=values.device, dtype=values.dtype)
    logits = (values @ scaling.unsqueeze(0) - absolute_differences.sum(dim=1, keepdim=True)).T
    permutation = torch.softmax(logits / temperature, dim=-1)
    positions = torch.arange(n - 1, -1, -1, device=values.device, dtype=values.dtype).unsqueeze(1)
    return (permutation.T @ positions).reshape(-1)



class MSELoss(ContextLoss):
    def forward(self, preds: Tensor, labels: Tensor, context=None) -> Tensor:
        preds, labels = finite_vectors(preds, labels)
        return F.mse_loss(preds, labels) if preds.numel() else preds.sum() * 0.0


class PearsonICLoss(ContextLoss):
    def forward(self, preds: Tensor, labels: Tensor, context=None) -> Tensor:
        return -weighted_corr(preds, labels)


class DifferentiableRankICLoss(ContextLoss):
    """Negative differentiable Spearman correlation on one daily cross-section."""

    def __init__(self, temperature: float = 0.1, method: Literal["sigmoid", "neural"] = "sigmoid", eps: float = 1e-8):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if method not in {"sigmoid", "neural"}:
            raise ValueError("method must be 'sigmoid' or 'neural'")
        self.temperature = float(temperature)
        self.rank_fn = sigmoid_rank if method == "sigmoid" else neural_sort_rank
        self.eps = eps

    def ranks(self, values: Tensor) -> Tensor:
        values = values.reshape(-1)
        values = (values - values.mean()) / values.std(unbiased=False).clamp_min(self.eps)
        return self.rank_fn(values, self.temperature)

    def correlation(self, first: Tensor, second: Tensor) -> Tensor:
        first, second = finite_vectors(first, second)
        if first.numel() < 2:
            return first.sum() * 0.0
        return weighted_corr(self.ranks(first), self.ranks(second), eps=self.eps)

    def forward(self, preds: Tensor, labels: Tensor, context=None) -> Tensor:
        return -self.correlation(preds, labels)


