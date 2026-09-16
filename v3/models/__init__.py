from __future__ import annotations

from importlib import import_module

__all__ = ["GRUConfig", "GRUModel", "GRU_Arg", "GRU_Model"]


def __getattr__(name):
    if name in __all__:
        module = import_module("v3.models.gru")
        return getattr(module, name)
    raise AttributeError(name)
