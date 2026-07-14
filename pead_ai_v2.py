# -*- coding: utf-8 -*-
"""
最终优化版：PIT 财报事件驱动横截面 Alpha 模型

核心优化：
1. 一个 batch = 一个交易日 trade_date 的全市场股票横截面。
2. market_x 走 Market Transformer。
3. event_x 是最近 K 条 PIT 可见事件记忆库，不是当天事件矩阵。
4. 时间距离不再用 embedding，而是作为连续特征进入 event_x：
       log1p(days_since_announce)
       log1p(days_since_report_period)
5. 删除冗余的 Event Attention Pooling。
   因为 Market -> Event Cross Attention 本身已经在做事件选择。
6. 第一层就判断无事件股票：
       has_event = event_mask.sum(dim=1) > 0
   无事件股票不跑 Event Transformer，不跑 Cross Attention，直接 fused_vec = market_vec。
7. 支持同一天多事件、多报告期、财报修正、PIT。
8. 训练目标以横截面排序为主：IC Loss + Pairwise Rank Loss + Huber Loss + Direction Loss。

输入：
    market_x:   [B, L, Dm]
    event_x:    [B, K, De]
    event_mask: [B, K]
    stock_mask: [B]
    y:          [B] 或 [B, H]

PIT 规则：
    事件进入 event memory 的唯一条件：effective_date <= trade_date
    不要使用 report_period <= trade_date
    A股保守：effective_date = next_trade_date(announce_date)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    market_dim: int
    event_dim: int
    hidden_dim: int = 128
    num_heads: int = 4
    market_layers: int = 2
    event_layers: int = 2
    dropout: float = 0.10
    num_targets: int = 1
    output_cls_logit: bool = True


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    return (x * mask).sum(dim=dim) / mask.sum(dim=dim).clamp_min(eps)


def pearson_ic_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-8) -> torch.Tensor:
    pred = pred.view(-1)
    target = target.view(-1)
    if mask is not None:
        m = mask.bool().view(-1)
        pred = pred[m]
        target = target[m]
    if pred.numel() <= 2:
        return pred.new_tensor(0.0)
    pred = pred - pred.mean()
    target = target - target.mean()
    ic = (pred * target).mean() / (pred.std(unbiased=False).clamp_min(eps) * target.std(unbiased=False).clamp_min(eps))
    return -ic


def multi_target_ic_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, target_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    if pred.dim() == 1:
        pred = pred.unsqueeze(-1)
    if target.dim() == 1:
        target = target.unsqueeze(-1)
    losses = torch.stack([pearson_ic_loss(pred[:, h], target[:, h], mask) for h in range(pred.size(1))])
    if target_weight is not None:
        w = target_weight.to(device=losses.device, dtype=losses.dtype)
        w = w / w.sum().clamp_min(1e-8)
        return (losses * w).sum()
    return losses.mean()


def pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, max_pairs: int = 20000, margin_scale: float = 1.0) -> torch.Tensor:
    if pred.dim() > 1:
        pred = pred[:, 0]
    if target.dim() > 1:
        target = target[:, 0]
    pred = pred.view(-1)
    target = target.view(-1)
    if mask is not None:
        m = mask.bool().view(-1)
        pred = pred[m]
        target = target[m]
    n = pred.numel()
    if n <= 2:
        return pred.new_tensor(0.0)
    num_pairs = min(max_pairs, n * (n - 1) // 2)
    i = torch.randint(0, n, (num_pairs,), device=pred.device)
    j = torch.randint(0, n, (num_pairs,), device=pred.device)
    diff_y = target[i] - target[j]
    valid = diff_y.abs() > 1e-12
    if valid.sum() == 0:
        return pred.new_tensor(0.0)
    i = i[valid]
    j = j[valid]
    sign = torch.sign(diff_y[valid])
    score = sign * (pred[i] - pred[j])
    return F.softplus(-margin_scale * score).mean()


def huber_regression_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if pred.dim() == 1:
        pred = pred.unsqueeze(-1)
    if target.dim() == 1:
        target = target.unsqueeze(-1)
    if mask is not None:
        m = mask.bool().view(-1)
        pred = pred[m]
        target = target[m]
    if pred.numel() == 0:
        return pred.new_tensor(0.0)
    return F.smooth_l1_loss(pred, target)


def direction_loss(logit: torch.Tensor, target_return: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if logit.dim() > 1:
        logit = logit[:, 0]
    if target_return.dim() > 1:
        target_return = target_return[:, 0]
    logit = logit.view(-1)
    label = (target_return.view(-1) > 0).float()
    if mask is not None:
        m = mask.bool().view(-1)
        logit = logit[m]
        label = label[m]
    if logit.numel() == 0:
        return logit.new_tensor(0.0)
    return F.binary_cross_entropy_with_logits(logit, label)


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class MarketEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(cfg.market_dim),
            nn.Linear(cfg.market_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.pos = SinusoidalPositionEmbedding(cfg.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.market_layers)
        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, market_x: torch.Tensor, market_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.pos(self.proj(market_x))
        key_padding_mask = None if market_mask is None else ~market_mask.bool()
        seq = self.encoder(x, src_key_padding_mask=key_padding_mask)
        seq = self.norm(seq)
        vec = seq[:, -1] if market_mask is None else masked_mean(seq, market_mask, dim=1)
        return seq, vec


class EventMemoryEncoder(nn.Module):
    """
    只处理有事件股票。
    无事件股票在 PITEventAlphaModel.forward 里直接跳过，不会进入本模块。
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.event_proj = nn.Sequential(
            nn.LayerNorm(cfg.event_dim),
            nn.Linear(cfg.event_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.event_layers)
        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, event_x: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
        x = self.event_proj(event_x)
        key_padding_mask = ~event_mask.bool()
        seq = self.encoder(x, src_key_padding_mask=key_padding_mask)
        seq = self.norm(seq)
        return seq


class MarketEventCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, market_vec: torch.Tensor, event_seq: torch.Tensor, event_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        query = market_vec.unsqueeze(1)
        key_padding_mask = ~event_mask.bool()
        context, attn = self.attn(
            query=query,
            key=event_seq,
            value=event_seq,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        context = context.squeeze(1)
        event_attn = attn.squeeze(1)
        gate = self.gate(torch.cat([market_vec, context, market_vec - context], dim=-1))
        fused = self.out_norm(gate * context + (1.0 - gate) * market_vec)
        return fused, event_attn


class PITEventAlphaModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.market_encoder = MarketEncoder(cfg)
        self.event_encoder = EventMemoryEncoder(cfg)
        self.cross_attention = MarketEventCrossAttention(cfg.hidden_dim, cfg.num_heads, cfg.dropout)
        self.alpha_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_targets),
        )
        self.cls_head = None
        if cfg.output_cls_logit:
            self.cls_head = nn.Sequential(
                nn.LayerNorm(cfg.hidden_dim),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim // 2, cfg.num_targets),
            )

    def forward(self, market_x: torch.Tensor, event_x: torch.Tensor, event_mask: torch.Tensor, market_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, K, _ = event_x.shape

        market_seq, market_vec = self.market_encoder(market_x, market_mask=market_mask)

        # 默认路径：所有股票先退化为纯行情路径。
        # 只有 has_event=True 的股票才会替换为融合事件后的 fused_vec。
        fused_vec = market_vec.clone()
        event_cross_attention = market_x.new_zeros(B, K)

        has_event = event_mask.sum(dim=1) > 0

        if has_event.any():
            idx = has_event.nonzero(as_tuple=False).squeeze(-1)
            event_seq_sub = self.event_encoder(event_x[idx], event_mask[idx])
            fused_sub, attn_sub = self.cross_attention(market_vec[idx], event_seq_sub, event_mask[idx])
            fused_vec[idx] = fused_sub
            event_cross_attention[idx] = attn_sub

        pred = self.alpha_head(fused_vec)
        if pred.size(-1) == 1:
            pred = pred.squeeze(-1)

        out = {
            "pred": pred,
            "market_vec": market_vec,
            "fused_vec": fused_vec,
            "has_event": has_event,
            "event_cross_attention": event_cross_attention,
        }

        if self.cls_head is not None:
            cls_logit = self.cls_head(fused_vec)
            if cls_logit.size(-1) == 1:
                cls_logit = cls_logit.squeeze(-1)
            out["cls_logit"] = cls_logit

        return out


def compute_loss(outputs: Dict[str, torch.Tensor], y: torch.Tensor, stock_mask: Optional[torch.Tensor] = None, target_weight: Optional[torch.Tensor] = None, w_ic: float = 0.60, w_pair: float = 0.25, w_reg: float = 0.10, w_cls: float = 0.05) -> Dict[str, torch.Tensor]:
    pred = outputs["pred"]
    loss_ic = multi_target_ic_loss(pred, y, stock_mask, target_weight)
    loss_pair = pairwise_rank_loss(pred, y, stock_mask)
    loss_reg = huber_regression_loss(pred, y, stock_mask)
    total = w_ic * loss_ic + w_pair * loss_pair + w_reg * loss_reg
    loss_cls = pred.new_tensor(0.0)
    if "cls_logit" in outputs and w_cls > 0:
        loss_cls = direction_loss(outputs["cls_logit"], y, stock_mask)
        total = total + w_cls * loss_cls
    return {
        "loss": total,
        "loss_ic": loss_ic.detach(),
        "loss_pair": loss_pair.detach(),
        "loss_reg": loss_reg.detach(),
        "loss_cls": loss_cls.detach(),
    }


EVENT_FEATURE_SCHEMA = """
event_x 每一条 event token 推荐包含：

一、事件类型类特征：
    event_type_onehot:
        年报 / 一季报 / 半年报 / 三季报 / 业绩预告 / 业绩快报 /
        财报修正 / 审计意见异常 / 分红送转 / 调研 / 回购 / 增减持 ...
    report_type_onehot:
        Q1 / Q2 / Q3 / Annual
    is_revision:
        是否财报修正
    revision_gap:
        修正后关键指标 - 修正前关键指标

二、财务数值特征：
    revenue_yoy
    net_profit_yoy
    deduct_net_profit_yoy
    roe
    gross_margin
    net_margin
    operating_cashflow_yoy
    debt_to_asset
    eps
    bps

三、surprise 特征：
    eps_surprise
    revenue_surprise
    profit_surprise
    forecast_upper_lower_mid
    actual_vs_forecast_mid

四、事件前市场反应：
    pre_5d_return
    pre_20d_return
    pre_20d_turnover
    pre_20d_volatility
    pre_20d_abnormal_volume

五、时间特征，作为连续变量：
    log1p(days_since_announce)
    log1p(days_since_report_period)
    days_since_announce / 252
    days_since_report_period / 365

六、PIT和质量控制：
    is_after_hours_announcement
    is_pre_open_announcement
    is_audited
    audit_opinion_type
    data_source_quality
"""


PIT_DATASET_PSEUDOCODE = r'''
# 原始事件表必须保持 long format，不要按 stock-date 聚合。
# events_long columns:
#   stock
#   announce_datetime
#   effective_date
#   event_type
#   report_period
#   report_type
#   is_revision
#   revision_target_period
#   financial_features
#   surprise_features
#   pre_event_market_features

def build_one_day_batch(trade_date, universe, L, K):
    for stock in universe:
        market_x = load_market_window(stock, trade_date, L)

        # PIT 可见事件记忆库。唯一准入条件：effective_date <= trade_date。
        ev = events_long[
            (events_long.stock == stock)
            & (events_long.effective_date <= trade_date)
        ].sort_values(["effective_date", "announce_datetime"]).tail(K)

        # 同一天多个事件不聚合，直接作为多条 token。
        # 例如：
        #   2009-04-30 发布 2008 年报
        #   2009-04-30 修正 2007 年报
        # 则：
        #   token0 = 2008 年报
        #   token1 = 2007 年报修正

        ev["log_days_since_announce"] = log1p(trade_day_diff(ev.effective_date, trade_date))
        ev["log_days_since_report_period"] = log1p(calendar_day_diff(ev.report_period, trade_date))

        event_x, event_mask = encode_and_pad_event_tokens(ev, K)
        y = future_return(stock, trade_date, H) - future_industry_return(stock, trade_date, H)
'''


def make_toy_batch(B: int = 6, L: int = 10, Dm: int = 8, K: int = 5, De: int = 18, num_targets: int = 3, seed: int = 42) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    market_x = torch.randn(B, L, Dm, generator=g) * 0.05
    event_x = torch.randn(B, K, De, generator=g) * 0.10

    valid_counts = torch.tensor([5, 3, 4, 0, 2, 5])[:B]
    if B > valid_counts.numel():
        valid_counts = torch.randint(0, K + 1, (B,), generator=g)

    event_mask = torch.zeros(B, K, dtype=torch.long)
    for i, c in enumerate(valid_counts):
        event_mask[i, : int(c)] = 1

    event_x = event_x * event_mask.unsqueeze(-1)
    event_x[:, :, 0:5] = 0.0

    # 第0只股票最近两条事件同一天发生：
    # token0：2008 年报；token1：2007 年报修正。
    if B >= 1 and K >= 2:
        event_x[0, 0, 0] = 1.0       # 年报
        event_x[0, 1, 3] = 1.0       # 财报修正
        event_x[0, 1, 5] = 1.0       # is_revision
        event_x[0, 0, 15] = torch.log1p(torch.tensor(1.0))
        event_x[0, 1, 15] = torch.log1p(torch.tensor(1.0))
        event_x[0, 0, 16] = torch.log1p(torch.tensor(120.0))
        event_x[0, 1, 16] = torch.log1p(torch.tensor(480.0))
        event_x[0, 0, 10] = 0.15
        event_x[0, 0, 11] = 0.08
        event_x[0, 1, 17] = -0.20

    for b in range(B):
        for k in range(K):
            if event_mask[b, k] == 0:
                continue
            if b == 0 and k < 2:
                continue
            event_type_idx = torch.randint(0, 5, (1,), generator=g).item()
            event_x[b, k, 0:5] = 0.0
            event_x[b, k, event_type_idx] = 1.0
            days_announce = torch.randint(1, 250, (1,), generator=g).float()
            days_period = torch.randint(30, 1200, (1,), generator=g).float()
            event_x[b, k, 15] = torch.log1p(days_announce)
            event_x[b, k, 16] = torch.log1p(days_period)

    event_x = event_x * event_mask.unsqueeze(-1)

    stock_mask = torch.ones(B, dtype=torch.long)
    if B >= 5:
        stock_mask[4] = 0

    if num_targets == 1:
        y = torch.tensor([0.034, -0.012, 0.008, -0.020, 0.000, 0.015], dtype=torch.float32)[:B]
        if B > 6:
            y = torch.randn(B, generator=g) * 0.03
    else:
        y = torch.randn(B, num_targets, generator=g) * 0.03
        if B >= 6 and num_targets >= 3:
            y = torch.tensor(
                [
                    [0.018, 0.034, 0.052],
                    [-0.006, -0.012, -0.025],
                    [0.004, 0.008, 0.011],
                    [-0.013, -0.020, -0.030],
                    [0.000, 0.000, 0.000],
                    [0.007, 0.015, 0.021],
                ],
                dtype=torch.float32,
            )

    return {"market_x": market_x, "event_x": event_x, "event_mask": event_mask, "stock_mask": stock_mask, "y": y}


def demo_run() -> None:
    batch = make_toy_batch(num_targets=3)
    cfg = ModelConfig(
        market_dim=batch["market_x"].shape[-1],
        event_dim=batch["event_x"].shape[-1],
        hidden_dim=64,
        num_heads=4,
        market_layers=2,
        event_layers=2,
        dropout=0.10,
        num_targets=batch["y"].shape[-1] if batch["y"].dim() == 2 else 1,
        output_cls_logit=True,
    )
    model = PITEventAlphaModel(cfg)
    out = model(batch["market_x"], batch["event_x"], batch["event_mask"])
    target_weight = torch.tensor([0.2, 0.3, 0.5]) if cfg.num_targets == 3 else None
    losses = compute_loss(out, batch["y"], batch["stock_mask"], target_weight=target_weight)
    print("market_x:", tuple(batch["market_x"].shape))
    print("event_x:", tuple(batch["event_x"].shape))
    print("event_mask:", tuple(batch["event_mask"].shape))
    print("has_event:", out["has_event"].tolist())
    print("y:", tuple(batch["y"].shape))
    print("pred:", tuple(out["pred"].shape))
    print("event_cross_attention:", tuple(out["event_cross_attention"].shape))
    print({k: float(v) for k, v in losses.items()})


if __name__ == "__main__":
    demo_run()
