# V3 Training Framework

`v3` is a self-contained training package for the end-to-end stock modeling workflow. It copies the dataset and V2 training capabilities into a package-local structure, then separates configuration, data, trainer implementations, experiment orchestration, models, and runnable job scripts.

The goal is to make new models, losses, datasets, and training frameworks easier to add without changing unrelated modules.

## Directory Layout

```text
v3/
  config/          BaseConfig and config merge utilities
  dataset/         Dataset backends, collate functions, feature config helpers
  training/        Losses, optimizers, metrics, and trainer implementations
  experiments/     Framework runner, bagging, gridsearch, result summaries
  models/          Model definitions and model-specific default configs
  scripts/         Explicit-parameter job entry files
  registry.py      Lightweight model registry
```

## Design Boundaries

- `v3.dataset` only knows how to read local data and return samples.
- `v3.training` only knows how to train, validate, predict, and compute losses.
- `v3.experiments` owns outer experiment logic such as `supervise`, `rolling`, `kfold`, `bagging`, and `gridsearch`.
- `v3.models` owns model architecture and model-specific default config.
- `v3.scripts` is thin glue for explicit function calls from notebooks, schedulers, or Python job files.

This keeps the trainer from knowing about grid search, keeps the model from knowing about rolling windows, and keeps data loading separate from experiment policy.

## GRU Entry Point

Use `v3.scripts.train_gru.run_gru` with explicit keyword arguments.

```python
from v3.scripts.train_gru import run_gru

pred, label, histories = run_gru(
    framework="kfold",
    loss="rankic",
    folds=5,
    loss_params={"temperature": 0.01, "method": "sigmoid"},
)
```

Available frameworks:

```python
framework="supervise"  # one train/valid/test split
framework="rolling"    # rolling date windows
framework="kfold"      # contiguous time-block cross validation
```

Available ensemble modes:

```python
ensemble="none"
ensemble="bagging"
ensemble="gridsearch"
```

Examples:

```python
# 5-fold RankIC training
run_gru(framework="kfold", loss="rankic", folds=5)

# Rolling training with industry-domain RankIC
run_gru(framework="rolling", loss="domain_industry")

# Seed bagging around kfold training
run_gru(framework="kfold", loss="rankic", ensemble="bagging", members=5, folds=5)

# Grid search over loss parameters
run_gru(
    framework="kfold",
    loss="rankic",
    ensemble="gridsearch",
    folds=5,
    grid={"temperature": [0.005, 0.01, 0.02], "method": ["sigmoid", "neural"]},
)
```

## Config Overrides

Pass nested dictionaries through `config_override`. The override is merged into the model default config.

```python
from v3.scripts.train_gru import period_config, run_gru

run_gru(
    framework="supervise",
    loss="ic",
    config_override={
        **period_config(
            train_start="2016-01-01",
            train_end="2021-12-31",
            valid_start="2022-01-01",
            valid_end="2022-12-31",
            test_start="2023-01-01",
            test_end="2023-12-31",
        ),
        "training": {"device": "cuda:0", "num_epoch": 50},
        "optimizer": {"optim_params": {"lr": 5e-4}},
    },
)
```

## Loss Configuration

Loss presets live in `v3.scripts.train_gru.LOSS_PRESETS`. You can use a preset name or pass a full loss config.

```python
run_gru(loss="temporal_rankic", loss_params={"turnover_rate": 0.2})

run_gru(
    loss={
        "name": "domain_rankic",
        "params": {
            "domain_type": "industry",
            "temperature": 0.01,
            "provider_params": {"axis_root": "Z:/axis", "mask_root": "Z:/mask"},
        },
    }
)
```

To add a new loss, implement it under `v3/training/losses/` and register it in `v3/training/losses/__init__.py`.

## Dataset Configuration

`v3.dataset.config` has small helpers for flexible local feature discovery.

```python
from v3.dataset.config import dataset_params, feature_block

params = dataset_params(
    shared={
        "start_date": "2013-01-01",
        "end_date": "2025-12-31",
        "label": "Y.10D.zcorr",
        "mode": "universe",
        "pool_name": None,
        "fix_stock": None,
        "sample_size": None,
        "nanflit_set": ["dailyset"],
    },
    blocks={
        "dailyset": feature_block("/data/xujiayi/end2end/GRU_new/", lag=20),
    },
)

run_gru(config_override={"training": {"dataset": {"params": params}}})
```

## Adding A New Model

Create a file under `v3/models/` with:

- a `BaseConfig` subclass containing default `training`, `model`, and `optimizer` config;
- a PyTorch `nn.Module` model class;
- optional `@register_model("name", config_class=YourConfig)` registration.

Then create a thin script under `v3/scripts/` that calls `v3.experiments.run_training`, `run_bagging`, or `run_gridsearch` with explicit keyword arguments.

## Outputs

All V3 outputs are written under:

```text
{args.training.perf_path}/{args.model.name}/v3/{run_name}/
```

Typical files include:

- `best_model.pth`
- `loss_history.csv`
- `alpha.csv`
- `label.csv`
- `alpha_rolling.csv`
- `alpha_kfold_ensemble.csv`
- `alpha_ensemble.csv`
- `summary.csv`

## Verification

Basic syntax verification:

```python
python -m compileall v3
```
