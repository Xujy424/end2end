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
from v3.config import merge_dict


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



LOSS_PRESETS = {
    "mse": {"name": "mse", "params": {}},
    "ic": {"name": "ic", "params": {}},
    "rankic": {"name": "rankic", "params": {"temperature": 0.01, "method": "sigmoid"}},
    "domain_rankic_index": {
        "name": "domain_rankic",
        "params": {
            "temperature": 0.01,
            "method": "sigmoid",
            "domain_type": "index",
            "domains": ["hs300", "zz500", "zz1000", "others"],
            "domain_weights": [0.3, 0.3, 0.3, 0.1],
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
    "register_domain_provider", "TemporalRankICLoss", "build_loss",
    "loss_config",
    "neural_sort_rank", "sigmoid_rank", "weighted_corr",
]





