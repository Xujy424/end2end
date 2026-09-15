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

## DataPool DataLoader Path

V3 now provides `v3.dataset.DataPoolDailyBatchDataset`, registered as:

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

This matches the RankIC training style where `batch_size=1` means one trading day.

Compared with the old `MultiBatchDataset` style, this path is usually faster and cleaner when data is stored as axis-aligned `.bin` files because:

- field shape and dtype are inferred once by `DataPool`;
- memmap handles are cached instead of reopened by each backend;
- date/tick axes are shared across every field;
- each training step materializes only the current daily cross-section and lag window;
- the collate function is nearly a no-op for `batch_size=1`, avoiding extra cat/unsqueeze work;
- `pin_memory=True` plus `non_blocking=True` keeps CPU-to-GPU transfer efficient.

Recommended loader settings for GPU training:

```python
config_override = {
    "training": {
        "dataset": {"name": "datapool_daily"},
        "batch_size": 1,
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
"specified_param_dict": {
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
