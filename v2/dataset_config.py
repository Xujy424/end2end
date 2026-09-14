from __future__ import annotations

from pathlib import Path


def infer_fields(data_path, *, include_suffix=".bin", exclude_prefixes=("Y",), exclude_contains=("tradable", "date", "tick", "mask")):
    root = Path(data_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(root)
    fields = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix != include_suffix:
            continue
        stem = path.stem
        if stem.startswith(tuple(exclude_prefixes)) or any(token in stem for token in exclude_contains):
            continue
        fields.append(stem)
    return fields


def make_feature_block(data_path, *, fields=None, lag=None, **extra):
    cfg = {"data_path": str(data_path)}
    cfg["fields"] = list(fields) if fields is not None else infer_fields(data_path)
    if lag is not None:
        cfg["lag"] = lag
    cfg.update(extra)
    return cfg


def update_dataset_features(args, feature_blocks):
    args.training.dataset.params.specified_param_dict = feature_blocks
    return args
