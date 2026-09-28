from torch import nn


class ContextLoss(nn.Module):
    requires_ordered_batches = False
    requires_epoch_update = False

    def configure_dataset(self, dataset) -> None:
        pass


from .domain import (
    DOMAIN_PROVIDERS, DomainProvider, DomainWeightedRankICLoss,
    IndustryDomainProvider, IndexDomainProvider, build_domain_provider,
    register_domain_provider,
)
from .rankic import (
    DifferentiableRankICLoss, MSELoss, PearsonICLoss,
    neural_sort_rank, sigmoid_rank, weighted_corr,
)
from .temporal import TemporalRankICLoss
from .pairwise import PairwiseRankLoss
from .prime import PrimeCombinedLoss
from v3.config import merge_dict


LOSS_REGISTRY = {
    "mse": MSELoss,
    "ic": PearsonICLoss,
    "pearson_ic": PearsonICLoss,
    "rankic": DifferentiableRankICLoss,
    "domain_rankic": DomainWeightedRankICLoss,
    "temporal_rankic": TemporalRankICLoss,
    "pairwise_rank": PairwiseRankLoss,
    "prime": PrimeCombinedLoss,
}

def build_loss(name: str, params=None):
    try:
        loss_class = LOSS_REGISTRY[name.lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown loss {name!r}; available: {sorted(LOSS_REGISTRY)}") from exc
    return loss_class(**dict(params or {}))



LOSS_PRESETS = {
    "mse": {"name": "mse", "params": {}},
    "ic": {"name": "ic", "params": {}},
    "rankic": {"name": "rankic", "params": {"temperature": 0.01, "method": "sigmoid"}},
    "pairwise_rank": {
        "name": "pairwise_rank",
        "params": {"top_fraction": 0.2, "temperature": 1.0, "max_pairs": 65536},
    },
    "prime": {
        "name": "prime",
        "params": {
            "ic_weight": 1.0,
            "rank_weight": 0.5,
            "pairwise_weight": 0.2,
            "physics_weight": 0.05,
            "vol_neutral_weight": 0.05,
        },
    },
    "domain_rankic_index": {
        "name": "domain_rankic",
        "params": {
            "temperature": 0.01,
            "method": "sigmoid",
            "domain_type": "index",
            "domains": ["hs300", "zz500", "zz1000", "others"],
            "domain_weights": [0.025, 0.025, 0.8, 0.15],
        },
    },
}


def loss_config(name, **params):
    try:
        config = LOSS_PRESETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown loss preset {name!r}; available: {sorted(LOSS_PRESETS)}") from exc
    return merge_dict(config, {"params": params} if params else None)



__all__ = [
    "ContextLoss", "DifferentiableRankICLoss", "DOMAIN_PROVIDERS", "DomainProvider",
    "DomainWeightedRankICLoss", "IndustryDomainProvider", "IndexDomainProvider",
    "LOSS_PRESETS", "LOSS_REGISTRY", "MSELoss", "PearsonICLoss", "build_domain_provider",
    "register_domain_provider", "TemporalRankICLoss", "PairwiseRankLoss",
    "PrimeCombinedLoss", "build_loss",
    "loss_config",
    "neural_sort_rank", "sigmoid_rank", "weighted_corr",
]





