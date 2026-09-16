from __future__ import annotations

from importlib import import_module

__all__ = ["GRUConfig", "GRUModel", "GRU_Arg", "GRU_Model"]


MODEL_DICT = {
    "gru": ("v3.models.gru", "GRU_Model", "GRU_Config"),
}

def get_model_config(name):
    key = str(name).lower()
    try:
        module_name, model, cfg = MODEL_DICT[key]
    except KeyError as exc:
        raise KeyError(f"Unknown strategy {name!r}; available: {sorted(MODEL_DICT)}") from exc
    return getattr(import_module(module_name), cfg)


def get_model(name):
    key = str(name).lower()
    try:
        module_name, model, cfg = MODEL_DICT[key]
    except KeyError as exc:
        raise KeyError(f"Unknown strategy {name!r}; available: {sorted(MODEL_DICT)}") from exc
    return getattr(import_module(module_name), model)


__all__ = ["MODEL_DICT", "get_model_config", "get_model"]
