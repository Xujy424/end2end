from __future__ import annotations

from v3.training.trainer import TRAINER_REGISTRY


def run_plain(args, model_class, *, framework, run_name=None, **kwargs):
    key = (framework or args.training.get("framework", "supervise")).lower()
    try:
        trainer_fn = TRAINER_REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"Unknown training framework {key!r}; available: {sorted(TRAINER_REGISTRY)}") from exc
    return trainer_fn(args, model_class, run_name=run_name or args.model.loss.name, **kwargs)
