from __future__ import annotations

import torch.nn as nn

from v3.config import BaseConfig
from v3.models.registry import register_model


D_FIELDS = [
    "close_zscore", "open_zscore", "high_zscore", "low_zscore", "logvolume_zscore", "turnover_zscore",
    "close_pct", "open_pct", "high_pct", "low_pct", "logvolume_pct", "turnover_pct",
    "close2open", "high2open", "low2open", "high2low", "high2close", "low2close",
]

M_FIELDS = ["close2dopen", "high2dopen", "low2dopen", "ppos", "volume_adj2rollmean", "amount2rollmean"]


GRU_Config = {
    "name": "gru",
    "params": {
        "input_size_d": len(D_FIELDS),
        "input_size_m": len(M_FIELDS),
        "hidden_size": 128,
        "num_layers": 4,
        "dropout": 0.5,
    },
}


@register_model("gru", config_class=GRU_Config)
class GRU_Model(nn.Module):
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






