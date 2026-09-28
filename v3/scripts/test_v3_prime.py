from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v3.dataset import PRIME_FEATURE_BLOCKS
from v3.scripts.test import main


if __name__ == "__main__":
    model_update = {
        "params": {
            "block_dims": {
                name: len(block["fields"])
                for name, block in PRIME_FEATURE_BLOCKS.items()
            },
            "hidden_size": 32,
            "energy_size": 16,
            "dropout": 0.3,
            "sequence_pool": "last",
            "input_noise_std": 0.03,
            "heat_penalty": 0.5,
        },
    }
    loss_update = {
        "params": {
            "ic_weight": 1.0,
            "rank_weight": 0.5,
            "pairwise_weight": 0.2,
            "physics_weight": 0.05,
            "vol_neutral_weight": 0.05,
            "direction_weight": 0.01,
            "temperature": 0.1,
            "top_fraction": 0.2,
            "min_samples": 10,
        },
    }
    dataset_update = {
        "params": {
            "dataset_config": {
                "nan_filter_blocks": list(PRIME_FEATURE_BLOCKS),
            },
            "feature_blocks": PRIME_FEATURE_BLOCKS,
        },
    }
    strategy_update = None
    training_update = None
    optimizer_update = None

    prediction, label, histories = main(
        model_name="prime",
        model_update=model_update,
        loss_name="prime",
        loss_update=loss_update,
        dataset_update=dataset_update,
        strategy_name="rolling",
        strategy_update=strategy_update,
        trainer_name="supervised",
        training_update=training_update,
        optimizer_update=optimizer_update,
    )
