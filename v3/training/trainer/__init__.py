from .supervised import SupervisedTrainerV3, run_supervise
from .rolling_supervised import get_rolling_windows, run_rolling_supervise
from .kfold_supervised import contiguous_kfold_indices, cross_sectional_zscore, run_kfold_supervise

TRAINER_REGISTRY = {
    "supervise": run_supervise,
    "basic": run_supervise,
    "single": run_supervise,
    "rolling": run_rolling_supervise,
    "rolling_supervise": run_rolling_supervise,
    "kfold": run_kfold_supervise,
    "cv": run_kfold_supervise,
    "cross_validation": run_kfold_supervise,
}

__all__ = [
    "SupervisedTrainerV3",
    "TRAINER_REGISTRY",
    "contiguous_kfold_indices",
    "cross_sectional_zscore",
    "get_rolling_windows",
    "run_kfold_supervise",
    "run_rolling_supervise",
    "run_supervise",
]
