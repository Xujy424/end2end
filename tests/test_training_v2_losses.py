import numpy as np
import torch
from types import SimpleNamespace

from training_v2.losses import (
    DifferentiableRankICLoss,
    DomainProvider,
    DomainWeightedRankICLoss,
    IndustryDomainProvider,
    IndexDomainProvider,
    PearsonICLoss,
    TemporalRankICLoss,
    neural_sort_rank,
    sigmoid_rank,
    register_domain_provider,
)
from main.cross_validation_v2 import contiguous_kfold_indices, cross_sectional_zscore


def test_rank_losses_prefer_correct_order_and_have_finite_gradients():
    labels = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    for method in ("sigmoid", "neural"):
        correct = labels.clone().requires_grad_(True)
        reversed_prediction = (-labels).clone().requires_grad_(True)
        loss = DifferentiableRankICLoss(temperature=0.1, method=method)
        correct_loss = loss(correct, labels)
        reversed_loss = loss(reversed_prediction, labels)
        assert correct_loss < reversed_loss
        correct_loss.backward()
        assert torch.isfinite(correct.grad).all()


def test_temporal_loss_rewards_persistent_predictions():
    labels = torch.arange(6, dtype=torch.float32)
    current = labels.clone().requires_grad_(True)
    indices = torch.arange(6)
    loss = TemporalRankICLoss(turnover_rate=0.2, temperature=0.1)
    persistent = loss(current, labels, {"current_indices": indices, "previous_preds": labels})
    reversing = loss(current, labels, {"current_indices": indices, "previous_preds": -labels})
    assert persistent < reversing


def test_pearson_ic_is_negative_one_for_perfect_prediction():
    values = torch.arange(10, dtype=torch.float32)
    assert torch.allclose(PearsonICLoss()(values, values), torch.tensor(-1.0), atol=1e-6)


def test_soft_rank_shapes():
    values = torch.tensor([3.0, 1.0, 2.0])
    assert sigmoid_rank(values).shape == values.shape
    assert neural_sort_rank(values).shape == values.shape


def test_contiguous_kfold_covers_every_observation_once():
    splits = contiguous_kfold_indices(11, 4)
    validation = np.concatenate([fold for _, fold in splits])
    assert sorted(validation.tolist()) == list(range(11))
    assert all(len(np.intersect1d(train, valid)) == 0 for train, valid in splits)


def test_cross_sectional_zscore():
    import pandas as pd

    frame = pd.DataFrame([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    normalized = cross_sectional_zscore(frame)
    assert np.allclose(normalized.mean(axis=1), 0.0)
    assert np.allclose(normalized.std(axis=1, ddof=0), 1.0)


class _TestDomainProvider(DomainProvider):
    def __init__(self, split=3):
        self.split = split

    def get(self, date, ticks):
        left = np.arange(len(ticks)) < self.split
        return {"left": left, "right": ~left}


def test_domain_loss_accepts_registered_provider_and_equal_weights():
    register_domain_provider("test", _TestDomainProvider)
    loss = DomainWeightedRankICLoss(
        domain_type="test", provider_params={"split": 3}, temperature=0.1
    )
    loss.configure_dataset(
        SimpleNamespace(dates=np.array(["2024-01-02"]), ticks=np.arange(6).astype(str))
    )
    labels = torch.arange(6, dtype=torch.float32)
    preds = labels.clone().requires_grad_(True)
    value = loss(preds, labels, {"date_idx": [0], "tick_idxs": np.arange(6)})
    assert value < 0
    value.backward()
    assert torch.isfinite(preds.grad).all()


def _write_axes(root):
    axis = root / "axis"
    axis.mkdir()
    np.save(axis / "dates.npy", np.array(["2024-01-02", "2024-01-03"]))
    np.save(axis / "stock_ticks.npy", np.array(["A", "B", "C", "D"]))
    return axis


def test_industry_domain_provider_aligns_date_and_tickers(tmp_path):
    axis = _write_axes(tmp_path)
    masks = tmp_path / "masks"
    masks.mkdir()
    np.array([[1, 1, 2, np.nan], [2, 1, 2, 1]], dtype=np.float64).tofile(
        masks / "industry.bin"
    )
    provider = IndustryDomainProvider(axis, masks)
    result = provider.get("2024-01-03", ["D", "missing", "A", "C"])
    assert result["1.0"].tolist() == [True, False, False, False]
    assert result["2.0"].tolist() == [False, False, True, True]


def test_index_domain_provider_builds_composite_domains(tmp_path):
    axis = _write_axes(tmp_path)
    masks = tmp_path / "masks"
    masks.mkdir()
    arrays = {
        "hs300": [[1, 0, 0, 0], [0, 0, 0, 0]],
        "zz500": [[0, 1, 0, 0], [0, 0, 0, 0]],
        "zz1000": [[0, 0, 1, 0], [0, 0, 0, 0]],
    }
    for name, values in arrays.items():
        np.asarray(values, dtype=bool).tofile(masks / f"{name}_mask.bin")
    provider = IndexDomainProvider(axis, masks)
    result = provider.get("2024-01-02", ["A", "B", "C", "D", "missing"])
    assert result["zz800"].tolist() == [True, True, False, False, False]
    assert result["zz1000"].tolist() == [False, False, True, False, False]
    assert result["others"].tolist() == [False, False, False, True, False]
