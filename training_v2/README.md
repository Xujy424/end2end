# Training V2

This package is a parallel implementation. It does not replace or modify the
legacy `training`, `dataset`, or `model_hub` packages.

## Layout

- `losses/`: loss implementations and the loss factory.
- `optimizers/`: optimizer, scheduler, and early-stopping factories.
- `trainer/`: the context-aware daily cross-section trainer.
- `main/cross_validation_v2.py`: contiguous KFold and z-score ensemble.
- `main/loss_experiment_v2.py`: comparison of all report losses.
- `gru_rankic_v2.py`: callable GRU experiment entry point.

All V2 loss functions use `forward(preds, labels, context=None)`. The trainer
supplies date and ticker context, so index membership and previous-period
predictions do not leak into model code or dataset code.

## Loss configuration

```python
override = {
    "training": {"prediction_lag": 5, "batch_size": 1},
    "model": {
        "loss": {
            "name": "temporal_rankic",
            "params": {"temperature": 0.01, "turnover_rate": 0.1},
        }
    },
}
```

Use `domain_rankic` with `axis_root` and `mask_root` overrides when the job is
not running on the Windows host that exposes the research data as `Z:`.

## Running from a job or notebook

```python
from gru_rankic_v2 import run_gru_loss_study

summary, outputs = run_gru_loss_study(
    folds=4,
    train_val_range=("2018-01-01", "2021-12-31"),
    prediction_range=("2022-01-01", "2026-03-31"),
)
```

The temporal loss forces chronological daily batches, aligns stocks by ticker,
and averages gradients across the epoch before one optimizer update. Other
losses retain ordinary per-batch optimizer updates.
