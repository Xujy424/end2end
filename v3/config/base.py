from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover
    OmegaConf = None


class ConfigNode(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def to_config(value):
    if isinstance(value, ConfigNode):
        return value
    if isinstance(value, Mapping):
        return ConfigNode({key: to_config(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_config(item) for item in value]
    return value


def merge_dict(base: Mapping[str, Any], override: Mapping[str, Any] | None = None):
    result = deepcopy(dict(base))
    for key, value in dict(override or {}).items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


class BaseConfig:
    def __init__(self, override: Mapping[str, Any] | None = None):
        cfg = merge_dict(self.default(), override)
        self.cfg = OmegaConf.create(cfg) if OmegaConf is not None else to_config(cfg)
        self.bind()

    def default(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def bind(self):
        self.training = self.cfg.training
        self.dataset = self.cfg.dataset
        self.loss = self.cfg.loss
        self.model = self.cfg.model
        self.optimizer = self.cfg.optimizer
        self.strategy = self.cfg.get("strategy", None)
        return self

    def rewrite(self, override: Mapping[str, Any]):
        if OmegaConf is not None:
            self.cfg = OmegaConf.merge(self.cfg, override)
        else:
            self.cfg = to_config(merge_dict(self.cfg, override))
        return self.bind()

    def clone(self):
        return deepcopy(self)
