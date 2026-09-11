"""GRU entry point for the parallel RankIC study framework.

Call ``run_gru_loss_study`` from a notebook or job launcher. Importing this file
does not start a long-running experiment.
"""

from main.loss_experiment_v2 import run_loss_comparison
from model_hub.RNNs.gru import GRU_Arg, GRU_Model


def run_gru_loss_study(config_override=None, **experiment_kwargs):
    args = GRU_Arg(config_override)
    return run_loss_comparison(args, GRU_Model, **experiment_kwargs)
