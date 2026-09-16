# V3 Training Framework

`v3` is a self-contained training package for the end-to-end stock modeling workflow. It keeps configuration, data, trainer implementations, training strategies, models, and runnable job scripts in package-local modules.

The goal is to make new models, losses, datasets, and training frameworks easier to add without changing unrelated modules.

## Directory Layout

```text
v3/
  config/          BaseConfig and config merge utilities
  dataset/         Dataset backends and feature config helpers
  training/        Losses, optimizers, metrics, trainers, and strategies
    trainer/       Learning modes: supervised and self-supervised
    strategy/      Plain, kfold, rolling, bagging, and gridsearch orchestration
  models/          Model definitions and model-specific default configs
  scripts/         Explicit-parameter job entry files
  registry.py      Lightweight model registry
```

## Design Boundaries

- `v3.dataset` only knows how to read local data and return samples.
- `v3.training.trainer` owns learning modes: `supervised` and `self_supervised`.
- `v3.training.strategy` owns outer training organization such as `plain`, `kfold`, `rolling`, `bagging`, and `gridsearch`.
- `v3.models` owns model architecture and model-specific default config.
- `v3.scripts.train_gru` is the GRU job entry point for explicit function calls from notebooks, schedulers, or Python job files.

This keeps the trainer from knowing about grid search, keeps the model from knowing about rolling windows, and keeps data loading separate from run policy.

## GRU Entry Point

Use `v3.scripts.train_gru.train_gru` for the configured GRU supervised run.

```python
from v3.scripts.train_gru import train_gru

pred, label, histories = train_gru()
```

The current script is intentionally one clear chain:

```python
train_gru.py -> training.strategy.plain.run_plain -> SupervisedTrainerV3
```

For other run layouts, call strategy modules explicitly:

```python
from v3.training.strategy.kfold import run_kfold
from v3.training.strategy.rolling import run_rolling
from v3.training.strategy.bagging import run_bagging
from v3.training.strategy.gridsearch import run_gridsearch
```

Examples:

```python
from v3.models.gru import GRUModel
from v3.scripts.train_gru import build_args
from v3.training.strategy.kfold import run_kfold

args = build_args()
pred, label, histories = run_kfold(
    args,
    GRUModel,
    train_val_range=("2016-01-01", "2024-12-31"),
    prediction_range=("2025-01-01", "2025-12-31"),
    folds=5,
)
```

## Config Overrides

Pass nested dictionaries through `config_override`. The override is merged into the model default config.

```python
from v3.scripts.train_gru import train_gru

train_gru(
    config_override={
        "training": {"device": "cuda:0", "num_epoch": 50},
        "optimizer": {"optim_params": {"lr": 5e-4}},
    },
)
```

## Loss Configuration

Loss configuration lives in the model config. `v3.scripts.train_gru.build_args` sets the current run to index-domain RankIC:

```python
args = build_args()
print(args.model.loss.name)
print(args.model.loss.params)
```

To add a new loss, implement it under `v3/training/losses/` and register it in `v3/training/losses/__init__.py`.

## Dataset Configuration

Dataset configuration is declared directly in each model config. There is no extra dataset config helper layer in V3.

```python
"dataset": {
    "name": "datapool_batch",
    "params": {
        "dataset_config": {
            "label": "Y.10D",
            "mode": "universe",
            "pool_name": None,
            "fix_stock": None,
            "sample_size": None,
            "nan_filter_blocks": ["dailyset"],
        },
        "feature_blocks": {
            "dailyset": {
                "kind": "daily",
                "data_path": "model_input/dGRU",
                "fields": ["close_zscore", "close_pct"],
                "lag": 20,
            },
            "minuteset": {
                "kind": "minute",
                "data_path": "m_essentials",
                "fields": ["close", "volume"],
            },
        },
    },
}
```

Date ranges are not part of model or dataset config. The script entry point passes ranges to the selected trainer.

## Adding A New Model

Create a file under `v3/models/` with:

- a `BaseConfig` subclass containing default `training`, `model`, and `optimizer` config;
- a PyTorch `nn.Module` model class;
- optional `@register_model("name", config_class=YourConfig)` registration.

Then add a model-specific script entry point or extend `v3/scripts/train_gru.py` with explicit keyword arguments.

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

## DataPool DataLoader Path

V3 now provides `v3.dataset.BatchDataset`, registered as:

```python
training.dataset.name = "datapool_daily"
```

It uses `v3.dataset.datapool.DataPool` as the physical data access layer and keeps field memmaps open for the life of the dataset. One `__getitem__` returns one daily stock cross-section:

```text
feats["dailyset"]: (stock, lag, feature)
label:             (stock,)
date_idx:          int
tick_idxs:         (stock,)
```

This matches the RankIC training style where one trainer step means one trading day.

Compared with the old `MultiBatchDataset` style, this path is usually faster and cleaner when data is stored as axis-aligned `.bin` files because:

- field shape and dtype are inferred once by `DataPool`;
- memmap handles are cached instead of reopened by each backend;
- date/tick axes are shared across every field;
- each training step materializes only the current daily cross-section and lag window;
- DataLoader returns dataset items directly, avoiding extra collation work;
- `pin_memory=True` plus `non_blocking=True` keeps CPU-to-GPU transfer efficient.

Recommended loader settings for GPU training:

```python
config_override = {
    "training": {
        "dataset": {"name": "datapool_daily"},
        "num_workers": 2,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }
}
```

Use fewer workers if the storage backend is a network drive or if random date access causes paging pressure. Use `num_workers=0` for debugging because errors are easier to read.

Memory behavior:

- The dataset does not load all fields into RAM. It keeps read-only memmap objects and relies on the OS page cache.
- Each batch copies only `(stock_count, lag, feature_count)` float32 data plus labels.
- GPU memory stays bounded by one daily cross-section, not by the full date range.
- For maximum transfer efficiency, keep tensors contiguous and use `pin_memory=True` in the DataLoader and `to(device, non_blocking=True)` in the trainer.

Future high-throughput improvements:

- Store model-ready features as one fused field `(date, feature, tick)` or `(date, tick, feature)` to reduce per-feature file reads.
- Add a small per-worker rolling date cache when training windows are strictly chronological.
- Use float16/bfloat16 feature storage for low-precision inputs if model quality is stable.
- Keep labels/masks as separate cheap memmaps and avoid converting them to DataFrames during training.

### Batch vs Flatten Dataset Modes

The DataPool torch layer supports both existing training conventions:

```python
training.dataset.name = "datapool_batch"    # same as datapool_daily
training.dataset.name = "datapool_flatten"
```

`datapool_batch` returns one trading day per sample. It is the right choice for cross-sectional losses such as IC, RankIC, domain RankIC, and temporal RankIC.

```text
feats["dailyset"]  -> (stock, lag, daily_feature)
feats["minuteset"] -> (stock, minute, minute_feature)
label              -> (stock,)
```

`datapool_flatten` returns one `(date, stock)` pair per sample. It is the right choice for ordinary point-wise supervised losses or larger mini-batches of independent stock-day samples.

```text
feats["dailyset"]  -> (lag, daily_feature)
feats["minuteset"] -> (minute, minute_feature)
label              -> (1,)
```

Feature block frequency is inferred from the block name or config:

```python
"feature_blocks": {
    "dailyset": {
        "kind": "daily",
        "data_path": "model_input/dGRU",
        "fields": ["close_zscore", "close_pct"],
        "lag": 20,
    },
    "minuteset": {
        "kind": "minute",
        "data_path": "m_essentials",
        "fields": ["close", "volume"],
    },
}
```

Daily fields are expected to be axis-aligned 2-D memmaps `(date, tick)`. Minute fields are expected to be 3-D memmaps `(date, minute, tick)`, which DataPool exposes as `(stock, minute, field)` for a daily cross-section.

