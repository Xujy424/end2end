from __future__ import annotations

from .ensembles import run_bagging, run_gridsearch
from .runner import result_dir, run_training, summarize_prediction


def run_plain(args, model_class, *, framework, loss_config, run_name=None, **kwargs):
    return run_training(
        args,
        model_class,
        framework=framework,
        loss_config=loss_config,
        run_name=run_name or loss_config["name"],
        **kwargs,
    )


def run_bagging_entry(args, model_class, *, framework, loss_config, members=5, **kwargs):
    return run_bagging(
        args,
        model_class,
        members=members,
        framework=framework,
        loss_config=loss_config,
        **kwargs,
    )


def run_gridsearch_entry(args, model_class, *, framework, loss_config, grid=None, **kwargs):
    return run_gridsearch(
        args,
        model_class,
        grid or {},
        framework=framework,
        base_loss_config=loss_config,
        **kwargs,
    )


ENSEMBLE_REGISTRY = {
    "none": run_plain,
    "bagging": run_bagging_entry,
    "gridsearch": run_gridsearch_entry,
}

__all__ = [
    "ENSEMBLE_REGISTRY",
    "result_dir",
    "run_bagging",
    "run_gridsearch",
    "run_training",
    "summarize_prediction",
]
