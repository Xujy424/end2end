from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v3.config import TemplateConfig, merge_dict
from v3.dataset import dataset_config
from v3.paths import DATA_ROOT
from v3.training.plots import plot_cumulative_ic, plot_group_return, plot_loss_history
from v3.training.strategy import get_strategy, strategy_config
from v3.models.gru import GRU_MODEL_CONFIG, GRU_FEATURE_BLOCKS
from v3.training.losses import loss_config
from v3.training.optimizers import optimizer_config


def build_args(
    *,
    model="gru",
    dataset="batch",
    loss="domain_rankic_index",
    optimizer="adamw",
    strategy="plain",
    model_params: Mapping[str, Any] | None = None,
    dataset_params: Mapping[str, Any] | None = None,
    loss_params: Mapping[str, Any] | None = None,
    optimizer_params: Mapping[str, Any] | None = None,
    strategy_params: Mapping[str, Any] | None = None,
    training_params: Mapping[str, Any] | None = None,
    config_override: Mapping[str, Any] | None = None,
):
    if model != "gru":
        raise KeyError("train_gru currently supports model='gru' only")

    model_cfg = merge_dict(GRU_MODEL_CONFIG, model_params)
    dataset_cfg = dataset_config(
        dataset,
        params=merge_dict(
            {"feature_blocks": GRU_FEATURE_BLOCKS},
            (dataset_params or {}).get("params", {}),
        ),
        **{key: value for key, value in dict(dataset_params or {}).items() if key != "params"},
    )
    loss_cfg = loss_config(loss, **dict(loss_params or {}))
    if loss_cfg["name"] == "domain_rankic":
        provider = loss_cfg.setdefault("params", {}).setdefault("provider_params", {})
        provider.setdefault("axis_root", DATA_ROOT / "axis")
        provider.setdefault("mask_root", DATA_ROOT / "stock/index/mask")
        provider.setdefault("ticks_file", "stock_ticks.npy")

    cfg = TemplateConfig(
        model=model_cfg,
        dataset=dataset_cfg,
        loss=loss_cfg,
        optimizer=optimizer_config(optimizer, **dict(optimizer_params or {})),
        strategy=strategy_config(strategy, **dict(strategy_params or {})),
        training=training_params,
        override=config_override,
    )
    return cfg


def train_gru(
    *,
    model="gru",
    dataset="batch",
    loss="domain_rankic_index",
    optimizer="adamw_default",
    strategy="plain",
    trainer="supervised",
    run_name=None,
    model_params: Mapping[str, Any] | None = None,
    dataset_params: Mapping[str, Any] | None = None,
    loss_params: Mapping[str, Any] | None = None,
    optimizer_params: Mapping[str, Any] | None = None,
    strategy_params: Mapping[str, Any] | None = None,
    training_params: Mapping[str, Any] | None = None,
    config_override: Mapping[str, Any] | None = None,
):
    args = build_args(
        model=model,
        dataset=dataset,
        loss=loss,
        optimizer=optimizer,
        strategy=strategy,
        model_params=model_params,
        dataset_params=dataset_params,
        loss_params=loss_params,
        optimizer_params=optimizer_params,
        strategy_params=strategy_params,
        training_params=training_params,
        config_override=config_override,
    )
    from v3.models.gru import GRUModel

    strategy_fn = get_strategy(args.strategy.name)
    prediction, label, histories = strategy_fn(
        args,
        GRUModel,
        trainer=trainer,
        run_name=run_name or _default_run_name(args),
        **dict(args.strategy.get("params", {})),
    )
    _save_plots(args, _plot_run_name(run_name or _default_run_name(args), args.strategy.name), prediction, label, histories)
    return prediction, label, histories


def _default_run_name(args):
    return f"{args.model.name}_{args.loss.name}_{args.strategy.name}"


def _plot_run_name(run_name, strategy):
    return f"{run_name}/rolling" if str(strategy).lower() == "rolling" else run_name


def _save_plots(args, run_name, prediction, label, histories):
    if label is None:
        return
    out_dir = Path(args.training.perf_path).expanduser() / args.model.name / "v3" / run_name

    group_ax, group_ret = plot_group_return(prediction, label, num_group=10, title="Test Group Return")
    group_ax.figure.tight_layout()
    group_ax.figure.savefig(out_dir / "test_group_return.png", dpi=160)
    group_ret.to_csv(out_dir / "test_group_return.csv")

    ic_ax, ic_frame = plot_cumulative_ic(prediction, label, title="Test Cumulative IC / RankIC")
    ic_ax.figure.tight_layout()
    ic_ax.figure.savefig(out_dir / "test_cumulative_ic_rankic.png", dpi=160)
    ic_frame.to_csv(out_dir / "test_cumulative_ic_rankic.csv")

    if histories:
        loss_ax, loss_frame = plot_loss_history(histories, title="Train / Valid Loss")
        loss_ax.figure.tight_layout()
        loss_ax.figure.savefig(out_dir / "loss_history.png", dpi=160)
        loss_frame.to_csv(out_dir / "loss_history_merged.csv", index=False)


if __name__ == "__main__":
    train_gru()
