# -*- coding: utf-8 -*-
"""
ERED: Earnings / Events-driven Return 模型。

本文件只保留模型和损失函数，不包含 dataset 和数据处理逻辑。
数据读取由 dataset.vanilla / dataset.multicompose / dataset.eventstore 负责；
原始长表到事件数组的处理由 1_get_data/model_specific/get_ered.py 负责。

模型输入：
    price_x     [N, L, D_price]   量价窗口，通常来自 dailyset
    event_x     [N, K, D_event]   最近 K 个财报事件特征，来自 eventvec
    event_mask  [N, K]            事件槽位 mask，来自 eventmask
    event_age   [N, K]            距事件生效日的交易日 age，来自 eventage
    static_x    [N, D_static]     可选静态特征

模型输出：
    pred        [N]               横截面股票 score / 未来收益预测
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def finite_mask(*xs: Tensor) -> Tensor:
    """返回所有输入都为有限值的位置。"""
    mask = torch.ones_like(xs[0], dtype=torch.bool)
    for x in xs:
        mask = mask & torch.isfinite(x)
    return mask


def make_age_bucket(age: Tensor) -> Tensor:
    """把事件 age 分桶为粗粒度 PEAD 影响周期。"""
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


# ---------------------------------------------------------------------------
# 模型模块
# ---------------------------------------------------------------------------


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
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


class PriceEncoder(nn.Module):
    """量价窗口编码器：LayerNorm + Linear + GRU + residual MLP。"""

    def __init__(self, price_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input = nn.Sequential(
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
        self.post = ResidualMLP(hidden_dim, hidden_dim * 2, dropout=dropout)

    def forward(self, price_x: Tensor) -> Tensor:
        h = self.input(price_x)
        out, _ = self.gru(h)
        return self.post(out[:, -1])


class EventEncoder(nn.Module):
    """财报事件编码器：事件投影 + age embedding + 可学习衰减 + attention。"""

    def __init__(
        self,
        event_dim: int,
        hidden_dim: int,
        max_age_for_decay: float = 60.0,
        dropout: float = 0.1,
    ):
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
        self.out = ResidualMLP(hidden_dim, hidden_dim * 2, dropout=dropout)

    def forward(
        self,
        event_x: Tensor,
        event_mask: Tensor,
        event_age: Tensor,
        price_repr: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        valid = (event_mask > 0) & (event_age >= 0)
        age = event_age.clamp_min(0).float()
        h = self.event_proj(event_x) + self.age_emb(make_age_bucket(event_age))

        # PEAD 直觉：财报影响会持续一段时间，但强度随 age 改变。
        age_norm = (age / self.max_age_for_decay).clamp(0.0, 5.0)
        log_age = torch.log1p(age) / math.log1p(self.max_age_for_decay)
        decay = self.decay(torch.stack([age_norm, log_age], dim=-1)).squeeze(-1)
        decay = decay * valid.float()

        q = self.query(price_repr).unsqueeze(1)
        k = self.key(h)
        v = self.value(h)
        score = (q * k).sum(dim=-1) / math.sqrt(k.shape[-1])
        score = score + torch.log(decay.clamp_min(1e-6))
        score = score.masked_fill(~valid, -1e9)

        no_event = valid.sum(dim=1) == 0
        attn = torch.softmax(score, dim=-1)
        attn = torch.where(no_event.unsqueeze(-1), torch.zeros_like(attn), attn)

        event_repr = torch.sum(attn.unsqueeze(-1) * v, dim=1)
        event_repr = self.out(event_repr)
        event_repr = torch.where(no_event.unsqueeze(-1), torch.zeros_like(event_repr), event_repr)
        return event_repr, attn, decay


class StaticEncoder(nn.Module):
    def __init__(self, static_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.static_dim = int(static_dim)
        if self.static_dim > 0:
            self.net = nn.Sequential(
                nn.LayerNorm(static_dim),
                nn.Linear(static_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.net = None

    def forward(self, static_x: Tensor, batch_size: int, device: torch.device) -> Tensor:
        if self.net is None:
            return torch.zeros(batch_size, 0, device=device)
        return self.net(static_x)


class EREDModel(nn.Module):
    """量价 + 基本面事件驱动的横截面选股模型。"""

    def __init__(
        self,
        price_dim: int,
        event_dim: int,
        static_dim: int = 0,
        hidden_dim: int = 128,
        price_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.price_encoder = PriceEncoder(price_dim, hidden_dim, price_layers, dropout)
        self.event_encoder = EventEncoder(event_dim, hidden_dim, dropout=dropout)
        self.static_encoder = StaticEncoder(static_dim, hidden_dim, dropout=dropout)

        fusion_dim = hidden_dim * 2 + (hidden_dim if static_dim > 0 else 0)
        self.event_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLP(hidden_dim * 2, hidden_dim * 4, dropout=dropout),
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
        price_x: Tensor,
        event_x: Tensor,
        event_mask: Tensor,
        event_age: Tensor,
        static_x: Optional[Tensor] = None,
        return_aux: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        n = price_x.shape[0]
        device = price_x.device
        if static_x is None:
            static_x = torch.zeros(n, 0, device=device)

        price_repr = self.price_encoder(price_x)
        event_repr, event_attn, event_decay = self.event_encoder(event_x, event_mask, event_age, price_repr)
        gate = self.event_gate(torch.cat([price_repr, event_repr], dim=-1))
        static_repr = self.static_encoder(static_x, batch_size=n, device=device)

        parts = [price_repr, gate * event_repr]
        if static_repr.shape[-1] > 0:
            parts.append(static_repr)

        pred = self.head(self.fusion(torch.cat(parts, dim=-1))).squeeze(-1)
        if not return_aux:
            return pred
        return pred, {
            "price_repr": price_repr,
            "event_repr": event_repr,
            "event_gate": gate,
            "event_attn": event_attn,
            "event_decay": event_decay,
        }


# ---------------------------------------------------------------------------
# 损失函数和指标
# ---------------------------------------------------------------------------


class DailyICLoss(nn.Module):
    """单个日度横截面的 1 - Pearson IC。"""

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
        ic = (pred * label).mean() / (
            pred.pow(2).mean().sqrt() * label.pow(2).mean().sqrt() + self.eps
        )
        return 1.0 - ic


class PairwiseRankLoss(nn.Module):
    """用于横截面选股的 pairwise logistic ranking loss。"""

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


class PEADLoss(nn.Module):
    """Huber 回归 + daily IC + pairwise rank 的混合目标。"""

    def __init__(
        self,
        huber_weight: float = 1.0,
        ic_weight: float = 0.1,
        rank_weight: float = 0.1,
        huber_delta: float = 1.0,
    ):
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


@torch.no_grad()
def daily_ic(pred: Tensor, label: Tensor, eps: float = 1e-8) -> Tensor:
    pred = pred.reshape(-1)
    label = label.reshape(-1)
    valid = finite_mask(pred, label)
    pred = pred[valid]
    label = label[valid]
    if pred.numel() < 2:
        return pred.new_tensor(float("nan"))
    pred = pred - pred.mean()
    label = label - label.mean()
    return (pred * label).mean() / (
        pred.pow(2).mean().sqrt() * label.pow(2).mean().sqrt() + eps
    )


@torch.no_grad()
def daily_rank_ic(pred: Tensor, label: Tensor) -> Tensor:
    pred = pred.reshape(-1)
    label = label.reshape(-1)
    valid = finite_mask(pred, label)
    pred = pred[valid]
    label = label[valid]
    if pred.numel() < 2:
        return pred.new_tensor(float("nan"))
    pred_rank = torch.argsort(torch.argsort(pred)).float()
    label_rank = torch.argsort(torch.argsort(label)).float()
    return daily_ic(pred_rank, label_rank)


__all__ = [
    "DailyICLoss",
    "EREDModel",
    "EventEncoder",
    "PEADLoss",
    "PairwiseRankLoss",
    "PriceEncoder",
    "daily_ic",
    "daily_rank_ic",
]
