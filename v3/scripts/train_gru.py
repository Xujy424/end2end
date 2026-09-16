from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v3.models.gru import GRUConfig, GRUModel
from v3.paths import DATA_ROOT
from v3.training.strategy.plain import run_plain
from v3.training.plots import plot_cumulative_ic, plot_group_return, plot_loss_history

ROOT = DATA_ROOT

TRAIN_RANGE = ("2016-01-01", "2023-12-31")
VALID_RANGE = ("2024-01-01", "2024-12-31")
TEST_RANGE = ("2025-01-01", "2025-12-31")


def build_args(
    *,
    perf_path="~/PycharmProjects/Models/XJY_end2end/0_result/",
    device="cuda:0",
    num_epoch: int | None = None,
    config_override: Mapping[str, Any] | None = None,
):
    override = {
        "training": {
            "device": device,
            "perf_path": perf_path,
            "dataset": {
                "name": "datapool_batch",
                "params": {
                    "dataset_config": {
                        "root": ROOT,              # 可选，DataPool 根目录；默认自动 ROOT
                        "asset": "stock",           # 可选，默认 stock
                        "label": "Y.10D",           # 标签字段，会映射到 stock/model_input/labels/Y.10D.bin
                        "mode": "universe",         # 股票池模式
                        "pool_name": None,          # mode="pool" 时使用，如 "zz800"
                        "fix_stock": None,          # mode="fix" 时指定股票列表
                        "sample_size": None,        # mode="sample" 时每日随机抽样数量
                        "nanflit_set": ["dailyset"] # 用哪些 feature block 做 NaN 过滤
                    },
                    "feature_blocks": {
                        "dailyset": {
                            "kind": "daily",
                            "data_path": "model_input/dGRU",
                            "fields": [
                                "close_zscore", "open_zscore", "high_zscore", "low_zscore", "logvolume_zscore", "turnover_zscore",
                                "close_pct", "open_pct", "high_pct", "low_pct", "logvolume_pct", "turnover_pct",
                                "close2open", "high2open", "low2open", "high2low", "high2close", "low2close",
                            ],
                            "lag": 20,
                        },
                        # "minuteset": {
                        #     "kind": "minute",
                        #     "data_path": "m_essentials",
                        #     "fields": ["close2dopen", "high2dopen", "low2dopen", "ppos", "volume_adj2rollmean", "amount2rollmean"],
                        # },
                    }
                },
            }
        },
        "model": {
            "loss": {
                "name": "mse",
                # "params": {
                #     "temperature": 0.01,
                #     "method": "sigmoid",
                #     "domain_type": "index",
                #     "domains": ["hs300", "zz500", "zz1000", "others"],
                #     "domain_weights": [0.025, 0.025, 0.8, 0.15],
                #     "provider_params": {
                #         "axis_root": ROOT / "axis",
                #         "mask_root": ROOT / "stock/index/mask",
                #         "ticks_file": "stock_ticks.npy",
                #     },
                # },
            }
        },
    }
    if num_epoch is not None:
        override["training"]["num_epoch"] = int(num_epoch)
    if config_override:
        override = _merge_dict(override, config_override)
    return GRUConfig(override)


def train_gru(
    *,
    perf_path="~/PycharmProjects/Models/XJY_end2end/0_result/",
    device="cuda:0",
    num_epoch: int | None = None,
    run_name="index_domain_rankic_supervise_2016_2025",
    config_override: Mapping[str, Any] | None = None,
):
    args = build_args(
        perf_path=perf_path,
        device=device,
        num_epoch=num_epoch,
        config_override=config_override,
    )
    prediction, label, histories = run_plain(
        args,
        GRUModel,
        trainer="supervised",
        run_name=run_name,
        train_range=TRAIN_RANGE,
        valid_range=VALID_RANGE,
        test_range=TEST_RANGE,
    )
    _save_plots(args, run_name, prediction, label, histories)
    return prediction, label, histories


def _save_plots(args, run_name, prediction, label, histories):
    out_dir = Path(args.training.perf_path).expanduser() / args.model.name / "v3" / run_name

    group_ax, group_ret = plot_group_return(
        prediction,
        label,
        num_group=10,
        title="Test Group Return: index-domain RankIC, Y.10D",
    )
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


def _merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]):
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


if __name__ == "__main__":
    train_gru()
