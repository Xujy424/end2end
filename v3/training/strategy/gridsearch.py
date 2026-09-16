from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from v3.training.metrics import IC, rankIC
from .plain import run_plain


def run_gridsearch(
    args,
    model_class,
    *,
    grid: Mapping[str, list[Any]] | None = None,
    strategy_fn=run_plain,
    run_name="gridsearch",
    **kwargs,
):
    rows, outputs = [], {}
    grid = grid or {}
    keys = list(grid)
    values_iter = itertools.product(*(grid[key] for key in keys)) if keys else [()]

    for values in values_iter:
        params = dict(zip(keys, values))
        run_args = copy.deepcopy(args)
        run_args.loss.params.update(params)
        alias = ",".join(f"{key}={value}" for key, value in params.items()) or "default"
        pred, label, hist = strategy_fn(
            run_args,
            model_class,
            run_name=f"{run_name}/{alias}",
            **kwargs,
        )
        summary = _summarize_prediction(pred, label) if label is not None else _summarize_histories(hist)
        rows.append({"run": alias, **params, **summary})
        outputs[alias] = {"prediction": pred, "label": label, "histories": hist}

    summary = pd.DataFrame(rows)
    sort_col = "rank_ic_mean" if "rank_ic_mean" in summary else "valid_loss_min"
    summary = summary.sort_values(sort_col, ascending=sort_col != "rank_ic_mean")
    summary.to_csv(_result_dir(args, run_name) / "summary.csv", index=False)
    return summary, outputs


def _result_dir(args, *parts) -> Path:
    path = Path(args.training.perf_path).expanduser() / args.model.name / "v3"
    for part in parts:
        path = path / str(part)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _summarize_prediction(prediction: pd.DataFrame, label: pd.DataFrame) -> dict[str, float]:
    pred_values, label_values = prediction.values, label.values
    rank_ic = rankIC(pred_values, label_values)
    pearson_ic = IC(pred_values, label_values)
    rank_std = np.nanstd(rank_ic)
    ic_std = np.nanstd(pearson_ic)
    return {
        "rank_ic_mean": float(np.nanmean(rank_ic)),
        "rank_ic_ir": float(np.nanmean(rank_ic) / rank_std) if rank_std else np.nan,
        "ic_mean": float(np.nanmean(pearson_ic)),
        "ic_ir": float(np.nanmean(pearson_ic) / ic_std) if ic_std else np.nan,
    }


def _summarize_histories(histories) -> dict[str, float]:
    frames = [item for item in histories if isinstance(item, pd.DataFrame) and not item.empty]
    if not frames:
        return {"valid_loss_min": np.nan, "train_loss_min": np.nan}
    frame = pd.concat(frames, ignore_index=True)
    return {
        "valid_loss_min": float(frame["valid_loss"].min()) if "valid_loss" in frame else np.nan,
        "train_loss_min": float(frame["train_loss"].min()) if "train_loss" in frame else np.nan,
    }
