from .supervised import SupervisedTrainerV3, run_supervise
from .unsupervised import UnsupervisedTrainerV3

TRAINER_REGISTRY = {
    "supervised": SupervisedTrainerV3,
    "supervise": SupervisedTrainerV3,
    "unsupervised": UnsupervisedTrainerV3,
}

__all__ = [
    "SupervisedTrainerV3",
    "TRAINER_REGISTRY",
    "run_supervise",
    "UnsupervisedTrainerV3",
]
