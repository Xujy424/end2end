import torch as th
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from functools import reduce
from typing import List, Dict, Tuple
import multiprocessing as mp
import copy
from pathlib import Path
import time

from training.trainer import *
from training.metrics import rankIC, IC, calc_group_ret



def run(trainer, args, model_class, **kwargs):
    trainer = trainer(args, model_class, **kwargs)
    pred_df, label_df = trainer.train()
    return pred_df, label_df 


def _proc_worker(trainer_type, args_obj, model_class, kwargs, member_idx, device, seed):
    try:
        args_local = copy.deepcopy(args_obj)
    except Exception:
        # fallback: shallow copy
        args_local = copy.copy(args_obj)

    # set device and seed per-member
    args_local.training.device = device
    args_local.training.seed = int(seed)

    # set per-member perf path
    base_perf = Path(args_local.training.perf_path).expanduser() 
    member_dir = base_perf / args_local.model.name / 'bagging'  / f'member_{member_idx}'
    member_dir.mkdir(parents=True, exist_ok=True)
    args_local.training.perf_path = str(member_dir)

    # run
    trainer = TRAINER_DICT[trainer_type]
    pred_df, label_df = run(trainer, args_local, model_class, **kwargs)
    # save member outputs for aggregator
    pred_df.to_csv(member_dir / f'alpha_member_{member_idx}.csv')
    label_df.to_csv(member_dir / f'label_member_{member_idx}.csv')


def bagging_parallel(num_bag, trainer_type, args, model_class, kwargs=None, n_gpus=None):
    kwargs = kwargs or {}
    n_gpus = n_gpus or th.cuda.device_count()
    if n_gpus <= 0: raise RuntimeError('No GPUs detected for parallel bagging')

    procs: List[mp.Process] = []
    seeds = [int(time.time()) % 10000 + i for i in range(num_bag)]

    for i in range(num_bag):
        gpu_id = i % n_gpus
        device = f'cuda:{gpu_id}'
        p = mp.get_context('spawn').Process(
            target=_proc_worker,
            args=(trainer_type, args, model_class, kwargs, i+1, device, seeds[i])
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # aggregate results from saved CSVs
    base_perf = Path(args.training.perf_path).expanduser()
    bag_root = base_perf / args.model.name / 'bagging'
    preds = []
    for i in range(num_bag):
        member_dir = bag_root / f'member_{i+1}'
        pred_file = member_dir / f'alpha_member_{i+1}.csv'
        pred_df = pd.read_csv(pred_file, index_col=0)
        preds.append(pred_df.values)
    stacked = np.stack(preds, axis=0)
    ensemble = np.nanmean(stacked, axis=0)
    ensemble_df = pd.DataFrame(ensemble, index=pred_df.index, columns=pred_df.columns)
    ensemble_df.to_csv(args.training.perf_path/ args.model.name / 'bagging'  /'ensemble_df.csv')

    label_df = pd.read_csv(member_dir/f'label_member_{i+1}.csv', index_col=0)
    label_df.to_csv(args.training.perf_path/ args.model.name / 'bagging'  /'label_df.csv')

    return ensemble_df, label_df