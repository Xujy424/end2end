from __future__ import annotations

import copy
import itertools
from typing import Any, Mapping

import pandas as pd

from v3.experiments.runner import result_dir, run_training, summarize_prediction
from v3.training.trainer import cross_sectional_zscore


def run_bagging(args, model_class, *, members=5, framework="supervise", loss_config=None, standardize=True, run_name="bagging", **kwargs):
    if members < 1:
        raise ValueError("members must be positive")
    predictions, labels, histories = [], None, []
    for member in range(1, members + 1):
        member_args = copy.deepcopy(args)
        member_args.training.seed = int(member_args.training.seed) + member - 1
        pred, labels, hist = run_training(
            member_args,
            model_class,
            framework=framework,
            loss_config=loss_config,
            run_name=f"{run_name}/member_{member:02d}",
            **kwargs,
        )
        predictions.append(cross_sectional_zscore(pred) if standardize else pred)
        histories.extend(hist)
    ensemble = sum(predictions) / len(predictions)
    out_dir = result_dir(args, run_name)
    ensemble.to_csv(out_dir / "alpha_ensemble.csv")
    labels.to_csv(out_dir / "label.csv")
    return ensemble, labels, histories


def run_gridsearch(args, model_class, grid: Mapping[str, list[Any]], *, framework="supervise", base_loss_config=None, run_name="gridsearch", **kwargs):
    rows, outputs = [], {}
    keys = list(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        params = dict(zip(keys, values))
        loss_config = copy.deepcopy(base_loss_config or {"name": args.model.loss.name, "params": dict(args.model.loss.get("params", {}))})
        loss_config.setdefault("params", {}).update(params)
        alias = ",".join(f"{key}={value}" for key, value in params.items()) or "default"
        pred, label, hist = run_training(
            args,
            model_class,
            framework=framework,
            loss_config=loss_config,
            run_name=f"{run_name}/{alias}",
            **kwargs,
        )
        rows.append({"run": alias, **params, **summarize_prediction(pred, label)})
        outputs[alias] = {"prediction": pred, "label": label, "histories": hist}
    summary = pd.DataFrame(rows).sort_values("rank_ic_mean", ascending=False)
    summary.to_csv(result_dir(args, run_name) / "summary.csv", index=False)
    return summary, outputs
