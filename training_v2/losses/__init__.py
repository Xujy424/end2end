from .domain import DomainWeightedRankICLoss, IndexDomainProvider
from .rankic import (
    ContextLoss, DifferentiableRankICLoss, MSELoss, PearsonICLoss,
    neural_sort_rank, sigmoid_rank, weighted_corr,
)
from .temporal import TemporalRankICLoss

LOSS_REGISTRY = {
    "mse": MSELoss,
    "ic": PearsonICLoss,
    "pearson_ic": PearsonICLoss,
    "rankic": DifferentiableRankICLoss,
    "domain_rankic": DomainWeightedRankICLoss,
    "temporal_rankic": TemporalRankICLoss,
}


def build_loss(name: str, params=None):
    try:
        loss_class = LOSS_REGISTRY[name.lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown loss {name!r}; available: {sorted(LOSS_REGISTRY)}") from exc
    return loss_class(**dict(params or {}))


__all__ = [
    "ContextLoss", "DifferentiableRankICLoss", "DomainWeightedRankICLoss",
    "IndexDomainProvider", "LOSS_REGISTRY", "MSELoss", "PearsonICLoss",
    "TemporalRankICLoss", "build_loss", "neural_sort_rank", "sigmoid_rank",
    "weighted_corr",
]
