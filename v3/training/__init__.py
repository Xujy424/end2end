TRAINING_PRESETS = {
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
}

__all__ = [
    "TRAINER_PRESETS",
    
]