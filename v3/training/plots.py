from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v3.training.metrics import IC, cal_alpha, rankIC


def plot_group_return(prediction: pd.DataFrame, label: pd.DataFrame, *, num_group=10, ax=None, title="Group return"):
    group_ret = cal_alpha(prediction, label, num_group=num_group).cumsum()
    ax = ax or plt.subplots(figsize=(12, 7))[1]
    group_ret.plot(ax=ax, grid=True)
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("cumulative demeaned return")
    return ax, group_ret


def plot_loss_history(histories, *, ax=None, title="Loss history"):
    frame = _history_frame(histories)
    ax = ax or plt.subplots(figsize=(10, 5))[1]
    if "run" in frame:
        for run, item in frame.groupby("run"):
            ax.plot(item["epoch"], item["train_loss"], alpha=0.45, label=f"{run} train")
            ax.plot(item["epoch"], item["valid_loss"], alpha=0.85, label=f"{run} valid")
    else:
        ax.plot(frame["epoch"], frame["train_loss"], label="train")
        ax.plot(frame["epoch"], frame["valid_loss"], label="valid")
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    return ax, frame


def plot_cumulative_ic(prediction: pd.DataFrame, label: pd.DataFrame, *, ax=None, title="Cumulative IC / RankIC"):
    dates = prediction.index
    ic = pd.Series(IC(prediction.values, label.values), index=dates, name="IC")
    rank_ic = pd.Series(rankIC(prediction.values, label.values), index=dates, name="RankIC")
    frame = pd.concat([ic, rank_ic], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0).cumsum()
    ax = ax or plt.subplots(figsize=(12, 6))[1]
    frame.plot(ax=ax, grid=True)
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("cumulative correlation")
    return ax, frame


def plot_result_dir(result_dir, *, prediction_file=None, label_file=None, output_dir=None, num_group=10):
    result_dir = Path(result_dir).expanduser()
    output_dir = Path(output_dir or result_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction = pd.read_csv(_pick_file(result_dir, prediction_file, "alpha*.csv"), index_col=0)
    label = pd.read_csv(_pick_file(result_dir, label_file, "label*.csv"), index_col=0)

    ax, group_ret = plot_group_return(prediction, label, num_group=num_group)
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "group_return.png", dpi=160)
    plt.close(ax.figure)

    ax, ic_frame = plot_cumulative_ic(prediction, label)
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "cumulative_ic_rankic.png", dpi=160)
    plt.close(ax.figure)

    loss_files = sorted(result_dir.rglob("loss_history.csv"))
    loss_frame = pd.DataFrame()
    if loss_files:
        ax, loss_frame = plot_loss_history(loss_files)
        ax.figure.tight_layout()
        ax.figure.savefig(output_dir / "loss_history.png", dpi=160)
        plt.close(ax.figure)

    group_ret.to_csv(output_dir / "group_return.csv")
    ic_frame.to_csv(output_dir / "cumulative_ic_rankic.csv")
    if not loss_frame.empty:
        loss_frame.to_csv(output_dir / "loss_history_merged.csv", index=False)

    return {"group_return": group_ret, "ic": ic_frame, "loss": loss_frame}


def _pick_file(base_dir: Path, explicit, pattern):
    if explicit is not None:
        return Path(explicit).expanduser()
    matches = sorted(base_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern!r} file found under {base_dir}")
    return matches[0]


def _history_frame(histories):
    if isinstance(histories, (str, Path)):
        histories = [histories]
    if isinstance(histories, pd.DataFrame):
        return histories

    frames = []
    for idx, item in enumerate(histories, start=1):
        if isinstance(item, (str, Path)):
            frame = pd.read_csv(item)
            frame["run"] = Path(item).parent.name
        else:
            frame = item.copy()
            frame["run"] = f"run_{idx:02d}"
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["epoch", "train_loss", "valid_loss"])
    return pd.concat(frames, ignore_index=True)
