"""GRU entry point for the parallel RankIC study framework.

Call ``run_gru_loss_study`` from a notebook or job launcher. Importing this file
does not start a long-running experiment.
"""

import argparse
from pathlib import Path

from main.loss_experiment_v2 import run_loss_comparison
from model_hub.RNNs.gru import GRU_Arg, GRU_Model

ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else Path('/data/shanghai/xujiayi/workflow/data/')



def run_gru_loss_study(config_override=None, **experiment_kwargs):
    args = GRU_Arg(config_override)
    return run_loss_comparison(args, GRU_Model, **experiment_kwargs)


def domain_loss_config(domain_type, *, domains=None, domain_weights=None, provider_params=None,
                       temperature=0.01, method="sigmoid", min_samples=2):
    """Create a domain loss config without coupling this entry point to its data source."""
    params = {
        "temperature": temperature,
        "method": method,
        "domain_type": domain_type,
        "provider_params": dict(provider_params or {}),
        "min_samples": min_samples,
    }
    if domains is not None:
        params["domains"] = list(domains)
    if domain_weights is not None:
        params["domain_weights"] = list(domain_weights)
    return {"name": "domain_rankic", "params": params}


def run_gru_domain_study(config_override=None, *, domain_types=("industry", "index"),
                         domain_options=None, **experiment_kwargs):
    """Compare selectable domain schemes; newly registered providers work unchanged."""
    options = {
        "industry": {
            "provider_params": {"axis_root": ROOT/"axis", "mask_root": ROOT/"stock/mask"},
        },
        "index": {
            "domains": ("hs300", "zz500", "zz1000", "others"),
            "domain_weights": (0.025, 0.8, 0.175),
            "provider_params": {"axis_root": ROOT/"axis", "mask_root": ROOT/"stock/index/mask"},
        },
    }
    for name, overrides in (domain_options or {}).items():
        options[name] = {**options.get(name, {}), **overrides}
    loss_configs = {
        f"domain_{domain_type}": domain_loss_config(domain_type, **options.get(domain_type, {}))
        for domain_type in domain_types
    }
    return run_gru_loss_study(config_override, loss_configs=loss_configs, **experiment_kwargs)


def run_gru_turnover_study(config_override=None, *, turnover_rates=(0.05, 0.1, 0.2),
                           temperature=0.01, method="sigmoid", **experiment_kwargs):
    """Compare RankIC with chronological prediction-turnover penalties."""
    loss_configs = {
        "rankic": {"name": "rankic", "params": {"temperature": temperature, "method": method}},
        **{
            f"turnover_{rate:g}": {
                "name": "temporal_rankic",
                "params": {"temperature": temperature, "method": method, "turnover_rate": rate},
            }
            for rate in turnover_rates
        },
    }
    return run_gru_loss_study(config_override, loss_configs=loss_configs, **experiment_kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description="GRU RankIC v2 experiment entry point")
    parser.add_argument("--study", choices=("all", "domain", "turnover"), default="all")
    parser.add_argument("--domain", action="append", dest="domains",
                        help="domain provider to test; repeat for multiple providers")
    parser.add_argument("--folds", type=int, default=4)
    parsed = parser.parse_args(argv)
    kwargs = {"folds": parsed.folds}
    if parsed.study == "domain":
        return run_gru_domain_study(
            domain_types=tuple(parsed.domains or ("industry", "index")), **kwargs
        )
    if parsed.study == "turnover":
        return run_gru_turnover_study(**kwargs)
    return run_gru_loss_study(**kwargs)


if __name__ == "__main__":
    main()
