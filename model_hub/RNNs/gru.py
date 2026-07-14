import torch.nn as nn
import torch as th
import os
from training.args import BaseArg


class GRU_Model(nn.Module):

    def __init__(self, input_size_d, input_size_m, hidden_size, num_layers, dropout):
        super(GRU_Model, self).__init__()
        self.hidden_size=hidden_size

        self.d_gru = nn.GRU(
            input_size_d,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # self.m_gru = nn.GRU(
        #     input_size_m,
        #     hidden_size,
        #     num_layers=num_layers,
        #     batch_first=True,
        #     dropout=dropout if num_layers > 1 else 0
        # )

        self.pred_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # [h, weighted_e]
            nn.LayerNorm(hidden_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size//2),
            nn.LayerNorm(hidden_size//2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, 1)
        )

    def forward(self, x):
        # dx, mx = x['dailyset'], x['minuteset']
        dx = x['dailyset']
        dh, _ = self.d_gru(dx)
        dh = dh[:, -1, :]
        # mx = x['minuteset']
        # mh, _ = self.m_gru(mx)
        # mh = mh[:, -1, :]
        #preds = self.pred_head(th.cat([dh,mh],dim=-1)).squeeze(-1)
        preds = self.pred_head(dh).squeeze(-1)
        return preds



class GRU_Arg(BaseArg):
    d_fields = [
        'close_zscore','open_zscore','high_zscore','low_zscore','logvolume_zscore','turnover_zscore',
        'close_pct','open_pct','high_pct','low_pct','logvolume_pct','turnover_pct',
        'close2open','high2open','low2open','high2low','high2close','low2close',
    ]
    m_fields = ['close2dopen','high2dopen','low2dopen','ppos','volume_adj2rollmean','amount2rollmean']

    def get_default_config(self):
        # 1. 定义默认配置（嵌套字典）
        return {
            "training": {
                "device": "cuda:4",
                "seed": 480,
                "num_epoch": 100,
                "batch_size": 1,
                "early_stop_patience": 3,  # 5
                "early_stop_delta": 0,     # 1e-4,
                "period": {
                    "train_start": "2013-01-01",
                    "train_end": "2022-12-31",
                    "valid_start": "2023-01-01",
                    "valid_end": "2023-12-31",
                    "test_start": "2024-01-01",
                    "test_end": "2024-12-31",
                },
                "dataset": {
                    'name':'batch',
                    'params':{
                        "shared_param_dict" : {
                            "start_date": "2013-01-01",
                            "end_date": "2025-12-31",
                            "label": "Yyeo.10D",
                            "mode": "universe",
                            "pool_name": None,
                            "fix_stock": None,
                            "sample_size": None,
                            "nanflit_set": ['dailyset','minuteset','timecode']
                        },
                        "specified_param_dict" :{
                            'dailyset': {
                                'data_path': '/data/xujiayi/xjy/research_factors/model_input/dGRU/',
                                'fields': self.d_fields, 
                                'lag': 20,
                            },
                            # 'minuteset': {
                            #     'data_path': '/data/xujiayi/xjy/m_field/',
                            #     'fields': self.m_fields,
                            # },
                        }
                    }
                },
                "multi_gpu": False,
                "available_gpu": [4,5],
                "main_gpu": 4,
                "amp": False,
                "deterministic": False,
                "perf_path": "~/PycharmProjects/Models/XJY_end2end/0_result/"
            },
            "model": {
                "name": 'gru',
                "params":{
                    "input_size_d": len(self.d_fields),
                    "input_size_m": len(self.m_fields),
                    "hidden_size": 128,
                    "num_layers": 4,
                    "dropout": 0.5,
                },
                "loss": {
                    'name': 'ic',
                    'params': {}
                }
            },
            "optimizer": {
                "name": "adamw",
                "optim_params": {
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "eps": 1e-8
                },
                "accumulation_steps": 1,
                "if_grad_norm": True,
                "max_grad_norm": 3.0,
                "if_lr_decay": True,
                "scheduler": "reduce_lr_on_plateau",
                "sched_params": {
                    "mode": "min",
                    "factor": 0.5,
                    "patience": 4
                },
                "warmup":{
                    'enabled': False,
                    'name':'linearlr',
                    'epoch': 5,
                    'start_lr': 1e-8
                }
            }
        }



if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd

    from training.trainer.basic_supervise import BasicSuperviseTrainer
    from training.trainer.rolling_supervise import RollingSuperviseTrainer
    from training.trainer import get_rolling_windows
    from training.metrics import rankIC, IC, calc_group_ret


    args_class, model_class = GRU_Arg, GRU_Model
    args = args_class()
    print(args.model.params)

    # trainer1 = BasicSuperviseTrainer(args, model_class)
    # trainer1.train(save_loss=True)
    # _, pred_df, label_df = trainer1.inference()
    # trainer1.plot_cumsumIC(pred_df, label_df, name='Inference')
    # trainer1.plot_group_ret(pred_df, label_df, name='Inference')


    window_params = {
        'start_dt': '2013-01-01',
        'end_dt': '2025-12-31',
        'train_len': 7,
        'valid_len': 1,
        'test_len': 1,
        'rolling_gap': 1,
    }
    windows,_ = get_rolling_windows(**window_params)

    trainer2 = RollingSuperviseTrainer(args, model_class, windows)
    trainer2.set_seed(args.training.seed)
    pred_df, label_df = trainer2.train()
    trainer2.plot_group_ret(pred_df, label_df, name='Merge')
    trainer2.plot_cumsumIC(pred_df, label_df, name='Merge')

    _, pred_df, label_df = trainer2.inference(
        date_range=(windows[0][2][0],windows[-1][2][1]),
    )
    trainer2.plot_group_ret(pred_df, label_df, name='Inference')
    trainer2.plot_cumsumIC(pred_df, label_df, name='Inference')
    

    # from main.bagging import bagging_parallel
    # from training.plots import plot_group_ret

    # window_params = {
    #     'start_dt': '2013-01-01',
    #     'end_dt': '2025-12-31',
    #     'train_len': 7,
    #     'valid_len': 1,
    #     'test_len': 1,
    #     'rolling_gap': 1,
    # }
    # windows,_ = get_rolling_windows(**window_params)

    # ensemble_df, label_df = bagging_parallel(
    #     5, 
    #     'rolling_supervise', 
    #     args, model_class,
    #     kwargs={'rolling_windows': windows}, 
    #     n_gpus=5)
    # plot_group_ret(ensemble_df, label_df, name='Bagging', perf_path=args.training)