import numpy as np
import torch

from training_v2.losses import (
    DifferentiableRankICLoss,
    PearsonICLoss,
    TemporalRankICLoss,
    neural_sort_rank,
    sigmoid_rank,
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
