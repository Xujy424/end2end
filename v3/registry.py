from __future__ import annotations

from collections.abc import Callable


class Registry(dict):
    def register(self, name: str, value=None):
        def decorator(obj):
            self[name] = obj
            return obj
        return decorator(value) if value is not None else decorator

    def build(self, name: str, *args, **kwargs):
        try:
            factory = self[name]
        except KeyError as exc:
            raise KeyError(f"Unknown registry key {name!r}; available: {sorted(self)}") from exc
        return factory(*args, **kwargs)


MODEL_REGISTRY = Registry()


def register_model(name: str, model_class=None, config_class=None):
    def decorator(cls):
        MODEL_REGISTRY[name] = {"model": cls, "config": config_class}
        return cls
    return decorator(model_class) if model_class is not None else decorator


def get_model_entry(name: str):
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}") from exc
