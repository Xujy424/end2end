from .supervised import SupervisedTrainerV3
from .self_supervised import BasicSelfSupervisedTrainerV3
from .multistage import MultiStageSupervisedTrainerV3
from .transfer import TransferSupervisedTrainerV3

TRAINER_REGISTRY = {
    "supervised": SupervisedTrainerV3,
    "self_supervised": BasicSelfSupervisedTrainerV3,
    "multistage": MultiStageSupervisedTrainerV3,
    "multi_stage": MultiStageSupervisedTrainerV3,
    "transfer": TransferSupervisedTrainerV3,
    "transfer_supervised": TransferSupervisedTrainerV3,
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
    "MultiStageSupervisedTrainerV3",
    "TransferSupervisedTrainerV3",
    "TRAINER_REGISTRY",
    "get_trainer_class",
]
