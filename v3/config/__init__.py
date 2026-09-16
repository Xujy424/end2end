from .base import BaseConfig, ConfigNode, merge_dict, to_config


BASE_CONFIG = {
    "trainer": {},
    "dataset": {},
    "loss": {},
    "model": {},
    "optimizer": {},
    "strategy": {},
}

__all__ = ["BASE_CONFIG", "BaseConfig", "ConfigNode", "TemplateConfig", "merge_dict", "to_config"]
