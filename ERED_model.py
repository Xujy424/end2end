"""财报事件驱动模型最终版。

输入沿用 ``ERED_model.py`` 的数据格式：

* dailyset:  [N, T, Dm]，日频市场特征；
* timecode:  DataEmbedding 所需的时间编码；
* eventvec:  [N, K, De]，按固定事件 ID 排列的事件特征；
* eventmask: [N, K]，有效事件为 1，无效事件为 0。

默认仅返回预测张量，以兼容现有监督训练器。需要分析事件权重时，可在
``forward`` 中传入 ``return_aux=True``，返回 ``(pred, aux)``。
"""

from typing import Dict, Optional, Tuple, Union
import os

import torch
import torch.nn as nn

from model_hub.Transformers.utils.Attn import AttentionLayer, FullAttention
from model_hub.Transformers.utils.Embed import DataEmbedding
from model_hub.Transformers.utils.EncDec import Encoder, EncoderLayer
from model_hub.Transformers.utils.masking import PaddingMask

from training.args import BaseArg


def masked_mean(
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """沿指定维度计算掩码均值；整行均无效时返回零向量。"""
    if mask is None:
        return x.mean(dim=dim)

    weight = mask.to(device=x.device, dtype=x.dtype)
    while weight.dim() < x.dim():
        weight = weight.unsqueeze(-1)
    numerator = (x * weight).sum(dim=dim)
    denominator = weight.sum(dim=dim).clamp_min(eps)
    return numerator / denominator


class MarketEncoder(nn.Module):
    """将日频市场序列编码为序列表示和样本级市场表示。"""

    def __init__(
        self,
        market_dim: int,
        hidden_dim: int,
        dropout: float,
        n_heads: int,
        d_ff: int,
        e_layers: int,
        embed_type: str = "fixed",
        freq: str = "d",
        activation: str = "gelu",
        output_attention: bool = False,
    ) -> None:
        super().__init__()
        self.market_proj = nn.Sequential(
            nn.LayerNorm(market_dim),
            nn.Linear(market_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.embedding = DataEmbedding(
            hidden_dim,
            hidden_dim,
            embed_type=embed_type,
            freq=freq,
            dropout=dropout,
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        hidden_dim,
                        n_heads,
                    ),
                    hidden_dim,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        market_x: torch.Tensor,
        timecode: torch.Tensor,
        market_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 非有限值在进入线性层前归零，避免一个坏值污染整条注意力序列。
        market_x = torch.nan_to_num(market_x, nan=0.0, posinf=0.0, neginf=0.0)
        hidden = self.embedding(self.market_proj(market_x), timecode)

        # padding mask 扩展后直接作为统一的 attn_mask。
        attn_mask = None if market_mask is None else PaddingMask(market_mask).mask
        sequence, _ = self.encoder(hidden, attn_mask=attn_mask)
        market_vector = (
            sequence[:, -1]
            if market_mask is None
            else masked_mean(sequence, market_mask, dim=1)
        )
        return sequence, market_vector


class EventSetEncoder(nn.Module):
    """编码固定事件槽位，并通过掩码注意力汇总有效事件。

    eventvec 的第 K 维对应固定的 event_id，因此事件 ID 嵌入能够保留不同
    财报字段组/事件槽位的身份；eventmask 则保证缺失槽位不参与池化。
    """

    def __init__(
        self,
        event_dim: int,
        hidden_dim: int,
        dropout: float,
        num_events: int,
    ) -> None:
        super().__init__()
        if num_events <= 0:
            raise ValueError("num_events 必须大于 0")

        self.num_events = num_events
        self.event_id_embedding = nn.Embedding(num_events, hidden_dim)
        self.event_projection = nn.Sequential(
            nn.LayerNorm(event_dim),
            nn.Linear(event_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.attention_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        event_x: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, num_events, _ = event_x.shape
        valid = event_mask.bool()
        event_x = torch.nan_to_num(event_x, nan=0.0, posinf=0.0, neginf=0.0)
        event_ids = torch.arange(num_events, device=event_x.device)
        hidden = self.event_projection(event_x)
        hidden = hidden + self.event_id_embedding(event_ids).unsqueeze(0)

        # 使用有限负数而非 -inf，兼容半精度训练和“整行无事件”的样本。
        scores = self.attention_score(hidden).squeeze(-1)   # N,K,hidden → N,K
        scores = scores.masked_fill(~valid, -1e4)
        weights = torch.softmax(scores, dim=-1) * valid.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        event_vector = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)

        no_event = ~valid.any(dim=1)
        event_vector = event_vector.masked_fill(no_event.unsqueeze(-1), 0.0)
        weights = weights.masked_fill(no_event.unsqueeze(-1), 0.0)
        return event_vector, weights


class EarningsReportEventDriven_Model(nn.Module):
    """市场基线 + 事件增量的财报事件驱动预测模型。"""

    REQUIRED_INPUTS = ("dailyset", "eventvec", "eventmask", "timecode")

    def __init__(
        self,
        num_target: int,
        market_dim: int,
        event_dim: int,
        hidden_dim: int,
        dropout: float,
        n_heads: int,
        d_ff: int,
        e_layers: int,
        num_events: int,
        embed_type: str = "fixed",
        freq: str = "d",
        activation: str = "gelu",
        output_attention: bool = False,
        # output_cls_logit: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim 必须能被 n_heads 整除")
        if hidden_dim < 2:
            raise ValueError("hidden_dim 必须不小于 2")

        # self.output_cls_logit = output_cls_logit
        self.market_encoder = MarketEncoder(
            market_dim,
            hidden_dim,
            dropout,
            n_heads,
            d_ff,
            e_layers,
            embed_type,
            freq,
            activation,
            output_attention,
        )
        self.event_encoder = EventSetEncoder(
            event_dim,
            hidden_dim,
            dropout,
            num_events,
        )

        # has_event 和 event_density 让门控显式知道事件是否存在及其稠密程度。
        self.event_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.market_head = self._make_head(hidden_dim, hidden_dim, num_target, dropout)
        self.event_delta_head = self._make_head(
            hidden_dim,
            hidden_dim // 2,
            num_target,
            dropout,
        )
        # self.cls_head = (
        #     self._make_head(hidden_dim, hidden_dim // 2, num_target, dropout)
        #     if output_cls_logit
        #     else None
        # )

    @staticmethod
    def _make_head(
        input_dim: int,
        inner_dim: int,
        output_dim: int,
        dropout: float,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, output_dim),
        )

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        event_mask = x["eventmask"].bool()

        _, market_vector = self.market_encoder(x["dailyset"], x["timecode"])
        event_vector, event_weights = self.event_encoder(x["eventvec"], event_mask)

        event_count = event_mask.sum(dim=1, keepdim=True).to(market_vector.dtype)
        has_event = (event_count > 0).to(market_vector.dtype)
        event_density = event_count / event_mask.size(1)
        gate_input = torch.cat(
            [market_vector, event_vector, has_event, event_density],
            dim=-1,
        )
        gate = self.event_gate(gate_input) * has_event
        fused_vector = self.fusion_norm(market_vector + gate * event_vector)

        # 无事件样本严格退化为市场基线；事件只学习相对基线的增量。
        prediction = self.market_head(market_vector)
        prediction = prediction + has_event * self.event_delta_head(fused_vector)
        if prediction.size(-1) == 1:
            prediction = prediction.squeeze(-1)

        if not return_aux:
            return prediction

        # auxiliary = {
        #     "has_event": has_event.squeeze(-1),
        #     "event_count": event_count.squeeze(-1),
        #     "event_weights": event_weights,
        #     "event_gate": gate,
        #     "market_vector": market_vector,
        #     "fused_vector": fused_vector,
        # }
        # if self.cls_head is not None:
        #     cls_logit = self.cls_head(fused_vector)
        #     auxiliary["cls_logit"] = (
        #         cls_logit.squeeze(-1) if cls_logit.size(-1) == 1 else cls_logit
        #     )
        return prediction   #, auxiliary



class EarningsReportEventDriven_Arg(BaseArg):
    d_fields = [
        'close_zscore','open_zscore','high_zscore','low_zscore','logvolume_zscore','turnover_zscore',
        'close_pct','open_pct','high_pct','low_pct','logvolume_pct','turnover_pct',
        'close2open','high2open','low2open','high2low','high2close','low2close',
    ]
    m_fields = ['close2dopen','high2dopen','low2dopen','ppos','volume_adj2rollmean','amount2rollmean']
    
    event_fields = [
        'basic_eps_yoy',
        'total_operating_revenue_yoy',
        'np_parent_company_owners_yoy',
        'net_operate_cash_flow_yoy',
        'roe_yoy','l2a_yoy','gpm_yoy','npm_yoy',
        # 'basic_eps_qoq',
        # 'total_operating_revenue_qoq',
        # 'np_parent_company_owners_qoq',
        # 'net_operate_cash_flow_qoq',
        # 'roe_qoq','l2a_qoq','gpm_qoq','npm_qoq',
        'eps_ue_sue', 'or_ue_sue', 'np_ue_sue', 'roe_ue_sue',
        'log_distance'
        'pct_20rollstd','pct_20rollmean','turnover_20rollstd','turnover_20rollmean'
    ]
    event_ids = [1,2,3,4]

    def get_default_config(self):
        # 1. 定义默认配置（嵌套字典）
        return {
            "training": {
                "device": "cuda:7",
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
                            "nanflit_set": ['dailyset','minuteset','timecode'],
                        },
                        "specified_param_dict" :{
                            'dailyset': {
                                'data_path': '/data/xujiayi/xjy/research_factors/model_input/dGRU/',
                                'fields': self.d_fields, 
                                'lag': 20,
                            },
                            # 'minuteset': {
                            #     'data_path': '/data/xujiayi/end2end/m_field/',
                            #     'fields': self.m_fields,
                            # },
                            'timecode': {
                                'freq': 'daily', # 'intraday'
                                'lag': 20,
                            },
                            'eventvec':{
                                'data_path': '/data/xujiayi/xjy/research_factors/model_input/ered/',
                                'event_ids': self.event_ids,
                                'fields': self.event_fields,
                            },
                            'eventmask':{
                                'data_path': '/data/xujiayi/xjy/research_factors/model_input/ered/',
                                'event_ids': self.event_ids,
                            }
                        }
                    }
                },
                "multi_gpu": False,
                "available_gpu": [6,7],
                "main_gpu": 7,
                "amp": False,
                "deterministic": False,
                "perf_path": "~/PycharmProjects/Models/XJY_end2end/0_result/"
            },
            "model": {
                "name": 'ered',
                "params":{
                    "num_target": 1,
                    "market_dim": len(self.d_fields),
                    "event_dim": len(self.event_fields),
                    "hidden_dim": 64,
                    "dropout": 0.5,
                    "n_heads": 4,
                    "d_ff": 128,
                    "e_layers": 2,
                    "num_events": len(self.event_ids),
                    "embed_type": "fixed",
                    "freq": "d",
                    "activation": "elu",
                    "output_attention": False,
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


    args_class, model_class = EarningsReportEventDriven_Arg, EarningsReportEventDriven_Model
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





