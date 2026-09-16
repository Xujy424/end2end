from .supervised import SupervisedTrainerV3
from .self_supervised import BasicSelfSupervisedTrainerV3

TRAINER_REGISTRY = {
    "supervised": SupervisedTrainerV3,
    "self_supervised": BasicSelfSupervisedTrainerV3,
}

def get_trainer_class(trainer=None, trainer_class=None):
    if trainer_class is not None:
        return trainer_class
    if trainer is None:
        return SupervisedTrainerV3
    if isinstance(trainer, str):
        try:
            return TRAINER_REGISTRY[trainer.lower()]
        except KeyError as exc:
            raise KeyError(f"Unknown trainer {trainer!r}; available: {sorted(TRAINER_REGISTRY)}") from exc
    return trainer


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
    "SupervisedTrainerV3",
    "BasicSelfSupervisedTrainerV3",
    "TRAINER_REGISTRY",
    "TRAINER_RESETS",
    
]
