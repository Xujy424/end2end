from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt

from v3.scripts.train_gru import ROOT, run_gru
from v3.training.metrics import cal_alpha


def run_index_domain_rankic_10d_return_supervised(
    *,
    perf_path="~/PycharmProjects/Models/XJY_end2end/0_result/",
    device="cuda:0",
    num_epoch=None,
    output_name="index_domain_rankic_10d_return_supervised_2016_2025",
):
    config_override = {
        "training": {
            "device": device,
            "perf_path": perf_path,
            "dataset": {
                "params": {
                    "dataset_config": {
                        "label": "Y.10D",
                    }
                }
            },
        }
    }
    if num_epoch is not None:
        config_override["training"]["num_epoch"] = int(num_epoch)

    prediction, label, histories = run_gru(
        framework="supervise",
        loss="domain_index",
        config_override=config_override,
        loss_params={
            "domains": ["zz800", "zz1000", "others"],
            "domain_weights": [0.025, 0.8, 0.175],
            "provider_params": {
                "axis_root": ROOT / "axis",
                "mask_root": ROOT / "stock/index/mask",
                "ticks_file": "stock_ticks.npy",
            },
        },
    )

    out_dir = Path(perf_path).expanduser() / "gru" / "v3" / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(out_dir / "test_prediction.csv")
    label.to_csv(out_dir / "test_label.csv")

    group_ret = cal_alpha(prediction, label, num_group=10).cumsum()
    group_ret.to_csv(out_dir / "test_group_return.csv")

    ax = group_ret.plot(grid=True, figsize=(12, 7), title="Test Group Return: index-domain RankIC, Y.10D")
    ax.set_xlabel("date")
    ax.set_ylabel("cumulative demeaned return")
    plt.tight_layout()
    fig_path = out_dir / "test_group_return.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()
    return prediction, label, group_ret, fig_path, histories


if __name__ == "__main__":
    run_index_domain_rankic_10d_return_supervised()



