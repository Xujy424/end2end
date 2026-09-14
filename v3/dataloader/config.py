from __future__ import annotations

from pathlib import Path
from typing import Iterable


def infer_fields(data_path, *, suffix=".bin", exclude_prefixes=("Y",), exclude_contains=("tradable", "date", "tick", "mask")):
    root = Path(data_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(root)
    fields = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix != suffix:
            continue
        stem = path.stem
        if stem.startswith(tuple(exclude_prefixes)) or any(token in stem for token in exclude_contains):
            continue
        fields.append(stem)
    return fields


def feature_block(data_path, *, fields: Iterable[str] | None = None, lag=None, **extra):
    cfg = {"data_path": str(data_path), "fields": list(fields) if fields is not None else infer_fields(data_path)}
    if lag is not None:
        cfg["lag"] = lag
    cfg.update(extra)
    return cfg


def dataset_params(shared, blocks):
    return {"shared_param_dict": dict(shared), "specified_param_dict": dict(blocks)}
