from __future__ import annotations

import copy
import itertools
from pathlib import Path

import pandas as pd

from training_v2.trainer import SupervisedTrainerV2, cross_sectional_zscore, run_kfold_supervise, run_rolling_supervise


def apply_loss_config(args, loss_config=None):
    run_args = copy.deepcopy(args)
    if loss_config:
        run_args.model.loss.name = loss_config.get("name", run_args.model.loss.name)
        run_args.model.loss.params = loss_config.get("params", {})
    return run_args


def run_supervise(args, model_class, *, loss_config=None, prediction_range=None, run_name=None):
    run_args = apply_loss_config(args, loss_config)
    trainer = SupervisedTrainerV2(run_args, model_class, run_name=run_name)
    history = trainer.fit()
    pred, label = trainer.predict(prediction_range, save=True)
    return pred, label, [history]


def run_training(args, model_class, *, framework="supervise", loss_config=None, run_name=None, **kwargs):
    framework = framework.lower()
    if framework in {"supervise", "basic", "single"}:
        return run_supervise(args, model_class, loss_config=loss_config, run_name=run_name, **kwargs)
    if framework in {"rolling", "rolling_supervise"}:
        return run_rolling_supervise(args, model_class, loss_config=loss_config, run_name=run_name, **kwargs)
    if framework in {"kfold", "cv", "cross_validation"}:
        return run_kfold_supervise(args, model_class, loss_config=loss_config, run_name=run_name, **kwargs)
    raise ValueError(f"Unknown framework {framework!r}")


def run_bagging(args, model_class, *, members=5, framework="supervise", loss_config=None, standardize=True, run_name="bagging", **kwargs):
    predictions, labels, histories = [], None, []
    for member in range(1, members + 1):
        member_args = copy.deepcopy(args)
        member_args.training.seed = int(member_args.training.seed) + member - 1
        pred, labels, hist = run_training(
            member_args, model_class, framework=framework, loss_config=loss_config,
            run_name=f"{run_name}/member_{member:02d}", **kwargs
        )
        predictions.append(cross_sectional_zscore(pred) if standardize else pred)
        histories.extend(hist)
    ensemble = sum(predictions) / len(predictions)
    out_dir = Path(args.training.perf_path).expanduser() / args.model.name / "v2" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ensemble.to_csv(out_dir / "alpha_ensemble.csv")
    labels.to_csv(out_dir / "label.csv")
    return ensemble, labels, histories


def run_gridsearch(args, model_class, grid, *, framework="supervise", base_loss_config=None, run_name="gridsearch", **kwargs):
    rows, outputs = [], {}
    keys = list(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        params = dict(zip(keys, values))
        loss_config = copy.deepcopy(base_loss_config or {"name": args.model.loss.name, "params": dict(args.model.loss.get("params", {}))})
        loss_config.setdefault("params", {}).update(params)
        alias = ",".join(f"{k}={v}" for k, v in params.items())
        pred, label, hist = run_training(args, model_class, framework=framework, loss_config=loss_config, run_name=f"{run_name}/{alias}", **kwargs)
        from main.loss_experiment_v2 import summarize_prediction
        rows.append({"run": alias, **params, **summarize_prediction(pred, label)})
        outputs[alias] = {"prediction": pred, "label": label, "histories": hist}
    summary = pd.DataFrame(rows).sort_values("rank_ic_mean", ascending=False)
    out_dir = Path(args.training.perf_path).expanduser() / args.model.name / "v2" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary, outputs
