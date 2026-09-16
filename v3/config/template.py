from __future__ import annotations

from typing import Any, Mapping

from v3.config.base import BaseConfig, merge_dict


BASE_CONFIG: dict[str, Any] = {
    "training": {
        "device": "cuda:0",
        "seed": 480,
        "num_epoch": 100,
        "early_stop_patience": 3,
        "early_stop_delta": 0,
        "num_workers": 0,
        "pin_memory": True,
        "persistent_workers": False,
        "prefetch_factor": 4,
        "multi_gpu": False,
        "available_gpu": [0],
        "main_gpu": 0,
        "amp": False,
        "deterministic": False,
        "perf_path": "~/PycharmProjects/Models/XJY_end2end/0_result/",
    },
    "dataset": {},
    "loss": {},
    "model": {},
    "optimizer": {},
    "strategy": {},
}


class TemplateConfig(BaseConfig):
    def __init__(
        self,
        *,
        dataset: Mapping[str, Any] | None = None,
        loss: Mapping[str, Any] | None = None,
        model: Mapping[str, Any] | None = None,
        optimizer: Mapping[str, Any] | None = None,
        strategy: Mapping[str, Any] | None = None,
        training: Mapping[str, Any] | None = None,
        override: Mapping[str, Any] | None = None,
    ):
        parts = {
            "dataset": dataset or {},
            "loss": loss or {},
            "model": model or {},
            "optimizer": optimizer or {},
            "strategy": strategy or {},
            "training": training or {},
        }
        super().__init__(merge_dict(parts, override))

    def default(self):
        return BASE_CONFIG
