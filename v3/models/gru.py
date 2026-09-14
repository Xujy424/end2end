from __future__ import annotations

import torch.nn as nn

from v3.config import BaseConfig
from v3.registry import register_model


class GRUConfig(BaseConfig):
    d_fields = [
        "close_zscore", "open_zscore", "high_zscore", "low_zscore", "logvolume_zscore", "turnover_zscore",
        "close_pct", "open_pct", "high_pct", "low_pct", "logvolume_pct", "turnover_pct",
        "close2open", "high2open", "low2open", "high2low", "high2close", "low2close",
    ]
    m_fields = ["close2dopen", "high2dopen", "low2dopen", "ppos", "volume_adj2rollmean", "amount2rollmean"]

    def default(self):
        return {
            "training": {
                "device": "cuda:4",
                "seed": 480,
                "num_epoch": 100,
                "batch_size": 1,
                "early_stop_patience": 3,
                "early_stop_delta": 0,
                "period": {
                    "train_start": "2013-01-01",
                    "train_end": "2022-12-31",
                    "valid_start": "2023-01-01",
                    "valid_end": "2023-12-31",
                    "test_start": "2024-01-01",
                    "test_end": "2024-12-31",
                },
                "dataset": {
                    "name": "batch",
                    "params": {
                        "shared_param_dict": {
                            "start_date": "2013-01-01",
                            "end_date": "2025-12-31",
                            "label": "y10_peer_zscore1",
                            "mode": "universe",
                            "pool_name": None,
                            "fix_stock": None,
                            "sample_size": None,
                            "nanflit_set": ["dailyset", "minuteset", "timecode"],
                        },
                        "specified_param_dict": {
                            "dailyset": {
                                "data_path": "/data/xujiayi/xjy/research_factors/model_input/dGRU/",
                                "fields": self.d_fields,
                                "lag": 20,
                            },
                        },
                    },
                },
                "multi_gpu": False,
                "available_gpu": [4, 5],
                "main_gpu": 4,
                "amp": False,
                "deterministic": False,
                "perf_path": "~/PycharmProjects/Models/XJY_end2end/0_result/",
            },
            "model": {
                "name": "gru",
                "params": {
                    "input_size_d": len(self.d_fields),
                    "input_size_m": len(self.m_fields),
                    "hidden_size": 128,
                    "num_layers": 4,
                    "dropout": 0.5,
                },
                "loss": {"name": "ic", "params": {}},
            },
            "optimizer": {
                "name": "adamw",
                "optim_params": {"lr": 1e-3, "weight_decay": 1e-4, "eps": 1e-8},
                "accumulation_steps": 1,
                "if_grad_norm": True,
                "max_grad_norm": 3.0,
                "if_lr_decay": True,
                "scheduler": "reduce_lr_on_plateau",
                "sched_params": {"mode": "min", "factor": 0.5, "patience": 4},
                "warmup": {"enabled": False, "name": "linearlr", "epoch": 5, "start_lr": 1e-8},
            },
        }


@register_model("gru", config_class=GRUConfig)
class GRUModel(nn.Module):
    def __init__(self, input_size_d, input_size_m, hidden_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_gru = nn.GRU(
            input_size_d,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        dx = x["dailyset"]
        dh, _ = self.d_gru(dx)
        dh = dh[:, -1, :]
        return self.pred_head(dh).squeeze(-1)


GRU_Arg = GRUConfig
GRU_Model = GRUModel
