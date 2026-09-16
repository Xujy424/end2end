from __future__ import annotations

from v3.config import merge_dict

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
