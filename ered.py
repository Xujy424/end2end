# -*- coding: utf-8 -*-
"""
ERED: Event-driven Return model.

模型逻辑：
- 量价特征是连续日频序列，由 MarketFeatureEncoder 编码；
- 基本面财报是稀疏真实事件，由 FundamentalEventEncoder 编码；
- 事件影响通过 event_age 的分桶和可学习 decay 表达 PEAD 式持续影响；
- 主干 EREDModel 使用门控融合，让模型决定当前横截面中基本面事件有多重要。

和 dataset.multicompose 的默认对接：
    feats['dailyset']   -> price_x     [N, L, D_price]
    feats['eventvec']   -> event_x     [N, K, D_event]
    feats['eventmask']  -> event_mask  [N, K]
    feats['eventage']   -> event_age   [N, K]
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from training.args import BaseArg


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------


def finite_mask(*xs: Tensor) -> Tensor:
    mask = torch.ones_like(xs[0], dtype=torch.bool)
    for x in xs:
        mask = mask & torch.isfinite(x)
    return mask


def make_age_bucket(age: Tensor) -> Tensor:
    """把事件 age 分成粗粒度 PEAD 持续影响区间。"""
    bucket = torch.zeros_like(age, dtype=torch.long)
    valid = age >= 0
    bucket = torch.where(valid & (age <= 1), torch.ones_like(bucket) * 1, bucket)
    bucket = torch.where(valid & (age >= 2) & (age <= 5), torch.ones_like(bucket) * 2, bucket)
    bucket = torch.where(valid & (age >= 6) & (age <= 10), torch.ones_like(bucket) * 3, bucket)
    bucket = torch.where(valid & (age >= 11) & (age <= 20), torch.ones_like(bucket) * 4, bucket)
    bucket = torch.where(valid & (age >= 21) & (age <= 40), torch.ones_like(bucket) * 5, bucket)
    bucket = torch.where(valid & (age >= 41) & (age <= 60), torch.ones_like(bucket) * 6, bucket)
    bucket = torch.where(valid & (age > 60), torch.ones_like(bucket) * 7, bucket)
    return bucket


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class MarketFeatureEncoder(nn.Module):
    def __init__(self, price_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(price_dim),
            nn.Linear(price_dim, hidden_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.post = ResidualMLP(hidden_dim, hidden_dim * 2, dropout)

    def forward(self, price_x: Tensor) -> Tensor:
        h = self.input_proj(price_x)
        out, _ = self.gru(h)
        return self.post(out[:, -1])


class FundamentalEventEncoder(nn.Module):
    def __init__(self, event_dim: int, hidden_dim: int = 128, max_age_for_decay: float = 60.0, dropout: float = 0.1):
        super().__init__()
        self.max_age_for_decay = float(max_age_for_decay)
        self.event_proj = nn.Sequential(
            nn.LayerNorm(event_dim),
            nn.Linear(event_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.age_emb = nn.Embedding(8, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.decay = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.post = ResidualMLP(hidden_dim, hidden_dim * 2, dropout)

    def forward(self, event_x: Tensor, event_mask: Tensor, event_age: Tensor, query_repr: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        valid = (event_mask > 0) & (event_age >= 0)
        age = event_age.clamp_min(0).float()
        h = self.event_proj(event_x) + self.age_emb(make_age_bucket(event_age))

        age_norm = (age / self.max_age_for_decay).clamp(0.0, 5.0)
        log_age = torch.log1p(age) / math.log1p(self.max_age_for_decay)
        decay = self.decay(torch.stack([age_norm, log_age], dim=-1)).squeeze(-1) * valid.float()

        q = self.query(query_repr).unsqueeze(1)
        k = self.key(h)
        v = self.value(h)
        score = (q * k).sum(dim=-1) / math.sqrt(k.shape[-1])
        score = score + torch.log(decay.clamp_min(1e-6))
        score = score.masked_fill(~valid, -1e9)

        no_event = valid.sum(dim=1) == 0
        attn = torch.softmax(score, dim=-1)
        attn = torch.where(no_event.unsqueeze(-1), torch.zeros_like(attn), attn)

        event_repr = torch.sum(attn.unsqueeze(-1) * v, dim=1)
        event_repr = self.post(event_repr)
        event_repr = torch.where(no_event.unsqueeze(-1), torch.zeros_like(event_repr), event_repr)
        return event_repr, {"event_attn": attn, "event_decay": decay, "event_valid": valid}


class ERED_Model(nn.Module):
    def __init__(
        self,
        price_dim: int,
        event_dim: int,
        hidden_dim: int = 128,
        price_layers: int = 2,
        dropout: float = 0.1,
        max_age_for_decay: float = 60.0,
    ):
        super().__init__()
        self.market_encoder = MarketFeatureEncoder(price_dim, hidden_dim, price_layers, dropout)
        self.event_encoder = FundamentalEventEncoder(
            event_dim=event_dim,
            hidden_dim=hidden_dim,
            max_age_for_decay=max_age_for_decay,
            dropout=dropout,
        )
        self.event_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        fusion_dim = hidden_dim * 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLP(hidden_dim * 2, hidden_dim * 4, dropout),
            nn.LayerNorm(hidden_dim * 2),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        price_x: Optional[Tensor] = None,
        event_x: Optional[Tensor] = None,
        event_mask: Optional[Tensor] = None,
        event_age: Optional[Tensor] = None,
        feats: Optional[Dict[str, Tensor]] = None,
        return_aux: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        """支持 model(batch['feats']) 和显式张量传参两种调用方式。"""
        if isinstance(price_x, dict) and feats is None:
            feats = price_x
            price_x = None

        if feats is not None:
            price_x = feats.get("dailyset", price_x)
            event_x = feats.get("eventvec", event_x)
            event_mask = feats.get("eventmask", event_mask)
            event_age = feats.get("eventage", event_age)

        if price_x is None or event_x is None or event_mask is None or event_age is None:
            raise ValueError("price_x/event_x/event_mask/event_age are required")

        price_repr = self.market_encoder(price_x)
        event_repr, event_aux = self.event_encoder(event_x, event_mask, event_age.long(), price_repr)
        gate = self.event_gate(torch.cat([price_repr, event_repr], dim=-1))

        parts = [price_repr, gate * event_repr]

        pred = self.head(self.fusion(torch.cat(parts, dim=-1))).squeeze(-1)
        if not return_aux:
            return pred
        aux = {"event_gate": gate, **event_aux}
        return pred, aux


class DailyICLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: Tensor, label: Tensor) -> Tensor:
        pred = pred.reshape(-1)
        label = label.reshape(-1)
        valid = finite_mask(pred, label)
        pred = pred[valid]
        label = label[valid]
        if pred.numel() < 2:
            return pred.new_tensor(1.0)
        pred = pred - pred.mean()
        label = label - label.mean()
        ic = (pred * label).mean() / (pred.pow(2).mean().sqrt() * label.pow(2).mean().sqrt() + self.eps)
        return 1.0 - ic


class PairwiseRankLoss(nn.Module):
    def __init__(self, max_pairs: int = 4096, margin_scale: float = 1.0):
        super().__init__()
        self.max_pairs = int(max_pairs)
        self.margin_scale = float(margin_scale)

    def forward(self, pred: Tensor, label: Tensor) -> Tensor:
        pred = pred.reshape(-1)
        label = label.reshape(-1)
        valid = finite_mask(pred, label)
        pred = pred[valid]
        label = label[valid]
        n = pred.numel()
        if n < 2:
            return pred.new_tensor(0.0)

        pair_count = min(self.max_pairs, n * (n - 1) // 2)
        i = torch.randint(0, n, (pair_count,), device=pred.device)
        j = torch.randint(0, n, (pair_count,), device=pred.device)
        neq = label[i] != label[j]
        if neq.sum() == 0:
            return pred.new_tensor(0.0)
        i = i[neq]
        j = j[neq]
        sign = torch.sign(label[i] - label[j])
        return F.softplus(-self.margin_scale * sign * (pred[i] - pred[j])).mean()


class ERED_Loss(nn.Module):
    def __init__(self, huber_weight: float = 1.0, ic_weight: float = 0.1, rank_weight: float = 0.1, huber_delta: float = 1.0):
        super().__init__()
        self.huber_weight = float(huber_weight)
        self.ic_weight = float(ic_weight)
        self.rank_weight = float(rank_weight)
        self.huber_delta = float(huber_delta)
        self.ic_loss = DailyICLoss()
        self.rank_loss = PairwiseRankLoss()

    def forward(self, pred: Tensor, label: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        pred = pred.reshape(-1)
        label = label.reshape(-1)
        valid = finite_mask(pred, label)
        pred_valid = pred[valid]
        label_valid = label[valid]
        if pred_valid.numel() == 0:
            zero = pred.sum() * 0.0
            return zero, {"huber": zero, "ic": zero, "rank": zero}

        huber = F.huber_loss(pred_valid, label_valid, delta=self.huber_delta)
        ic = self.ic_loss(pred_valid, label_valid)
        rank = self.rank_loss(pred_valid, label_valid)
        loss = self.huber_weight * huber + self.ic_weight * ic + self.rank_weight * rank
        return loss, {"huber": huber.detach(), "ic": ic.detach(), "rank": rank.detach()}



class ERED_Arg(BaseArg):
    d_fields = [
        'close_zscore','open_zscore','high_zscore','low_zscore','logvolume_zscore','turnover_zscore',
        'close_pct','open_pct','high_pct','low_pct','logvolume_pct','turnover_pct',
        'close2open','high2open','low2open','high2low','high2close','low2close',
    ]
    event_fields = [
        'ROETTM', 'ROICTTM', 'GrossIncomeRatioTTM', 'NetProfitRatioTTM',
        'PeriodCostsRateTTM', 'AdminiExpenseRateTTM',
        'TotalAssetTRateTTM', 'ARTRate', 'InventoryTRate',
        'DebtAssetsRatio', 'LongDebtRatio',
        'NPParentCompanyCutYOY', 'TotalAssetGrowRate', 'NetOperateCashFlowYOY',
        'NOCFToOperatingNITTM', 'SaleServiceCashToORTTM', 'OperCashInToAsset',
        'FixAssetRatio', 'IntangibleAssetRatio',
    ]

    def get_default_config(self):
        return {
            "training": {
                "device": "cuda:4",
                "seed": 480,
                "num_epoch": 100,
                "batch_size": 1,
                "early_stop_patience": 3,
                "early_stop_delta": 0,
                "period": {
                    "train_start": "2013-01-01",
                    "train_end": "2022-12-31",
                    "valid_start": "2023-01-01",
                    "valid_end": "2023-12-31",
                    "test_start": "2024-01-01",
                    "test_end": "2024-12-31",
                },
                "dataset": {
                    "name": "batch",
                    "params": {
                        "shared_param_dict": {
                            "start_date": "2013-01-01",
                            "end_date": "2025-12-31",
                            "label": "Yyeo.10D",
                            "mode": "universe",
                            "pool_name": None,
                            "fix_stock": None,
                            "sample_size": None,
                            # 只用 dailyset 做 NaN 过滤；事件分支有 padding，不参与过滤。
                            "nanflit_set": ["dailyset"],
                        },
                        "specified_param_dict": {
                            "dailyset": {
                                "data_path": "/data/xujiayi/xjy/research_factors/model_input/dGRU/",
                                "fields": self.d_fields,
                                "lag": 20,
                            },
                            "eventvec": {
                                "data_path": "/data/xujiayi/xjy/research_factors/model_specific/ered_v2/",
                                "max_events": 8,
                                "include_today": True,
                            },
                            "eventmask": {
                                "data_path": "/data/xujiayi/xjy/research_factors/model_specific/ered_v2/",
                                "max_events": 8,
                                "include_today": True,
                            },
                            "eventage": {
                                "data_path": "/data/xujiayi/xjy/research_factors/model_specific/ered_v2/",
                                "max_events": 8,
                                "include_today": True,
                            },
                        },
                    },
                },
                "multi_gpu": False,
                "available_gpu": [4, 5],
                "main_gpu": 4,
                "amp": False,
                "deterministic": False,
                "perf_path": "~/PycharmProjects/Models/XJY_end2end/0_result/",
            },
            "model": {
                "name": "ered",
                "params": {
                    "price_dim": len(self.d_fields),
                    "event_dim": len(self.event_fields),
                    "hidden_dim": 128,
                    "price_layers": 2,
                    "dropout": 0.3,
                    "max_age_for_decay": 60.0,
                },
                "loss": {
                    # 保持和现有训练框架一致；若 loss registry 支持 ERED_Loss，可改成 ered。
                    "name": "ic",
                    "params": {},
                },
            },
            "optimizer": {
                "name": "adamw",
                "optim_params": {
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "eps": 1e-8,
                },
                "accumulation_steps": 1,
                "if_grad_norm": True,
                "max_grad_norm": 3.0,
                "if_lr_decay": True,
                "scheduler": "reduce_lr_on_plateau",
                "sched_params": {
                    "mode": "min",
                    "factor": 0.5,
                    "patience": 4,
                },
                "warmup": {
                    "enabled": False,
                    "name": "linearlr",
                    "epoch": 5,
                    "start_lr": 1e-8,
                },
            },
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


    args_class, model_class = ERED_Arg, ERED_Model
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

    trainer2 = RollingSuperviseTrainer(args, model_class, windows, special_loss=ERED_Loss)
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








