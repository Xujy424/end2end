from __future__ import annotations

import copy
from pathlib import Path

from .plain import run_plain
from v3.training.trainer import cross_sectional_zscore


def run_bagging(args, model_class, *, members=5, framework="supervise", standardize=True, run_name="bagging", **kwargs):
    if members < 1:
        raise ValueError("members must be positive")
    predictions, labels, histories = [], None, []
    for member in range(1, members + 1):
        member_args = copy.deepcopy(args)
        member_args.training.seed = int(member_args.training.seed) + member - 1
        pred, labels, hist = run_plain(
            member_args,
            model_class,
            framework=framework,
            run_name=f"{run_name}/member_{member:02d}",
            **kwargs,
        )
        predictions.append(cross_sectional_zscore(pred) if standardize else pred)
        histories.extend(hist)
    ensemble = sum(predictions) / len(predictions)
    out_dir = _result_dir(args, run_name)
    ensemble.to_csv(out_dir / "alpha_ensemble.csv")
    labels.to_csv(out_dir / "label.csv")
    return ensemble, labels, histories


def _result_dir(args, *parts) -> Path:
    path = Path(args.training.perf_path).expanduser() / args.model.name / "v3"
    for part in parts:
        path = path / str(part)
    path.mkdir(parents=True, exist_ok=True)
    return path
