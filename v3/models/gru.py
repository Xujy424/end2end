from __future__ import annotations

import torch.nn as nn

from v3.dataset import DAILY_FIELDS, MINUTE_FIELDS
from v3.models.registry import register_model

GRU_Config = {
    "name": "gru",
    "params": {
        "daily_input_size": len(DAILY_FIELDS),
        "minute_input_size": len(MINUTE_FIELDS),
        "hidden_size": 128,
        "num_layers": 4,
        "dropout": 0.5,
    },
}

@register_model("gru", config_class=GRU_Config)
class GRU_Model(nn.Module):
    def __init__(self, daily_input_size, minute_input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.daily_input = nn.Linear(daily_input_size, hidden_size)
        self.minute_input = nn.Linear(minute_input_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size,
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
        if "minuteset" in x:
            sequence = self.minute_input(x["minuteset"])
        else:
            sequence = self.daily_input(x["dailyset"])
        hidden, _ = self.gru(sequence)
        return self.pred_head(hidden[:, -1, :]).squeeze(-1)
