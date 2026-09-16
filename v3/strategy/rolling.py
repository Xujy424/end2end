from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from v3.trainer import resolve_trainer_class


def get_rolling_windows(start_dt, end_dt, train_len=7, valid_len=1, test_len=1, rolling_gap=1):
    start = pd.Timestamp(start_dt)
    end = pd.Timestamp(end_dt)
    windows = []
    train_start = start
    while True:
        train_end = train_start + pd.DateOffset(years=train_len) - pd.Timedelta(days=1)
        valid_start = train_end + pd.Timedelta(days=1)
        valid_end = valid_start + pd.DateOffset(years=valid_len) - pd.Timedelta(days=1)
        test_start = valid_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(years=test_len) - pd.Timedelta(days=1)
        if test_start > end:
            break
        if test_end > end:
            test_end = end
        windows.append(
            (
                (str(train_start.date()), str(train_end.date())),
                (str(valid_start.date()), str(valid_end.date())),
                (str(test_start.date()), str(test_end.date())),
            )
        )
        train_start += pd.DateOffset(years=rolling_gap)
    return windows


def run_rolling(
    args,
    model_class,
    *,
    trainer="supervised",
    trainer_class=None,
    rolling_windows=None,
    window_params=None,
    run_name=None,
):
    trainer_class = resolve_trainer_class(trainer, trainer_class)
    if rolling_windows is None:
        if window_params is None:
            raise ValueError("run_rolling requires rolling_windows or window_params")
        rolling_windows = get_rolling_windows(**window_params)

    predictions, labels, histories = [], [], []
    base_name = run_name or args.loss.name
    for idx, (train_win, valid_win, test_win) in enumerate(rolling_windows, start=1):
        fold_args = copy.deepcopy(args)
        trainer = trainer_class(fold_args, model_class, run_name=f"{base_name}/rolling/window_{idx:02d}")
        histories.append(trainer.fit(train_range=train_win, valid_range=valid_win))
        pred, label = trainer.predict(test_win, save=True)
        predictions.append(pred)
        labels.append(label)
    pred_df = pd.concat(predictions).sort_index() if predictions else pd.DataFrame()
    label_df = pd.concat(labels).sort_index() if labels else pd.DataFrame()
    out_dir = Path(args.training.perf_path).expanduser() / args.model.name / "v3" / base_name / "rolling"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "alpha_rolling.csv")
    label_df.to_csv(out_dir / "label_rolling.csv")
    return pred_df, label_df, histories
