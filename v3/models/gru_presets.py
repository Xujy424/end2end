from __future__ import annotations

D_FIELDS = [
    "close_zscore", "open_zscore", "high_zscore", "low_zscore", "logvolume_zscore", "turnover_zscore",
    "close_pct", "open_pct", "high_pct", "low_pct", "logvolume_pct", "turnover_pct",
    "close2open", "high2open", "low2open", "high2low", "high2close", "low2close",
]

M_FIELDS = ["close2dopen", "high2dopen", "low2dopen", "ppos", "volume_adj2rollmean", "amount2rollmean"]

GRU_MODEL_CONFIG = {
    "name": "gru",
    "params": {
        "input_size_d": len(D_FIELDS),
        "input_size_m": len(M_FIELDS),
        "hidden_size": 128,
        "num_layers": 4,
        "dropout": 0.5,
    },
}

GRU_FEATURE_BLOCKS = {
    "dailyset": {
        "kind": "daily",
        "data_path": "model_input/dGRU",
        "fields": D_FIELDS,
        "lag": 20,
    },
}
