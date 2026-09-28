from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v3.dataset import PRIME_FEATURE_BLOCKS, PRIME_FIELDS
from v3.losses import PrimeCombinedLoss, build_loss
from v3.models import get_model, get_model_config
from v3.trainer.supervised import SupervisedTrainerV3


def _features(stocks=32, lag=1):
    generator = torch.Generator().manual_seed(7)
    return {
        name: torch.randn(stocks, lag, len(fields), generator=generator)
        for name, fields in PRIME_FIELDS.items()
    }


def test_prime_feature_blocks_are_explicit_and_match_model_dimensions():
    config = get_model_config("prime")
    dimensions = config["params"]["block_dims"]
    assert set(PRIME_FEATURE_BLOCKS) == set(dimensions)
    for name, block in PRIME_FEATURE_BLOCKS.items():
        assert block["fields"] == PRIME_FIELDS[name]
        assert len(block["fields"]) == dimensions[name]


def test_prime_model_and_loss_complete_a_backward_pass():
    model_class = get_model("prime")
    model = model_class(**get_model_config("prime")["params"])
    output = model(_features())
    predictions, structured = SupervisedTrainerV3._unpack_model_output(output)

    assert predictions.shape == (32,)
    assert set(structured["components"]) >= {
        "energy", "E_bull", "E_bear", "E_heat",
        "alpha_market", "beta_risk", "gamma_heat",
    }
    loss = build_loss("prime", {"min_samples": 4})
    assert isinstance(loss, PrimeCombinedLoss)
    value = loss(predictions, torch.randn(32), {"model_output": structured})
    assert torch.isfinite(value)
    value.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_prime_rejects_missing_or_wrong_sized_blocks():
    model = get_model("prime")(**get_model_config("prime")["params"])
    features = _features()
    del features["macro"]
    try:
        model(features)
    except KeyError as error:
        assert "macro" in str(error)
    else:
        raise AssertionError("missing macro block was accepted")

    features = _features()
    features["bull"] = torch.randn(32, 1, 15)
    try:
        model(features)
    except ValueError as error:
        assert "bull" in str(error)
    else:
        raise AssertionError("wrong bull dimension was accepted")


def test_supervised_trainer_runs_one_prime_optimization_step():
    trainer = SupervisedTrainerV3.__new__(SupervisedTrainerV3)
    trainer.device = torch.device("cpu")
    trainer.model = get_model("prime")(**get_model_config("prime")["params"])
    trainer.loss = build_loss("prime", {"min_samples": 4})
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)
    trainer.args = SimpleNamespace(
        optimizer=SimpleNamespace(if_grad_norm=False),
        training={"prediction_lag": 1},
    )
    batch = {
        "feats": _features(),
        "label": torch.randn(32),
        "date_idx": 0,
        "tick_idxs": torch.arange(32).numpy(),
    }
    loader = DataLoader([batch], batch_size=None)
    value = trainer._iterate(loader, training=True)
    assert torch.isfinite(torch.tensor(value))


def main():
    test_prime_feature_blocks_are_explicit_and_match_model_dimensions()
    test_prime_model_and_loss_complete_a_backward_pass()
    test_prime_rejects_missing_or_wrong_sized_blocks()
    test_supervised_trainer_runs_one_prime_optimization_step()
    print("PRIME v3 smoke tests passed")


if __name__ == "__main__":
    main()
