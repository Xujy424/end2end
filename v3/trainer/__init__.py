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


__all__ = [
    "SupervisedTrainerV3",
    "BasicSelfSupervisedTrainerV3",
    "TRAINER_REGISTRY",
]
