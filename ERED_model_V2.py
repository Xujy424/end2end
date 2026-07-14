# -*- coding: utf-8 -*-
"""
PIT ERED event-broadcast Transformer model.

This model is aligned with build_ered_event_dataset.py.

Core idea
---------
Financial reports are sparse, low-frequency events. They should not be used only
at the announcement day. The data pipeline keeps each report as a PIT-visible
long-format event, and the Dataset returns:

    market_x:    [B, L, Dm]
    event_x:     [B, K, De]
    event_mask:  [B, K]
    event_age:   [B, L, K]

where event_age[b, t, k] is the trading-day distance from the t-th market day
inside the lookback window to the k-th PIT-visible financial event.

The model broadcasts event information to every market time step through
content attention + learnable time decay. The final prediction is

    pred = base_pred + event_scale * event_residual

so the event branch learns the incremental contribution over the ordinary
market Transformer instead of replacing it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# @dataclass
# class ModelConfig:
#     market_dim: int
#     event_dim: int
#     hidden_dim: int = 128
#     num_heads: int = 4
#     market_layers: int = 2
#     event_layers: int = 1
#     dropout: float = 0.10
#     num_targets: int = 1
#     max_effect_days: int = 120
#     output_cls_logit: bool = False
#     event_scale_init: float = 0.05


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    return (x * mask).sum(dim=dim) / mask.sum(dim=dim).clamp_min(eps)


# def _as_2d(x: torch.Tensor) -> torch.Tensor:
#     if x.dim() == 1:
#         return x.unsqueeze(-1)
#     return x


# def weighted_pearson_ic_loss(
#     pred: torch.Tensor,
#     target: torch.Tensor,
#     mask: Optional[torch.Tensor] = None,
#     weight: Optional[torch.Tensor] = None,
#     eps: float = 1e-8,
# ) -> torch.Tensor:
#     pred = pred.view(-1)
#     target = target.view(-1)
#     valid = torch.isfinite(pred) & torch.isfinite(target)
#     if mask is not None:
#         valid = valid & mask.bool().view(-1)
#     pred = pred[valid]
#     target = target[valid]
#     if pred.numel() <= 2:
#         return pred.new_tensor(0.0)
#     if weight is None:
#         w = torch.ones_like(pred) / pred.numel()
#     else:
#         w = weight.view(-1)[valid].to(dtype=pred.dtype, device=pred.device)
#         w = w.clamp_min(0)
#         w = w / w.sum().clamp_min(eps)
#     pred_centered = pred - (pred * w).sum()
#     target_centered = target - (target * w).sum()
#     cov = (w * pred_centered * target_centered).sum()
#     pred_std = torch.sqrt((w * pred_centered.square()).sum().clamp_min(eps))
#     target_std = torch.sqrt((w * target_centered.square()).sum().clamp_min(eps))
#     return -(cov / (pred_std * target_std).clamp_min(eps))


# def multi_target_ic_loss(
#     pred: torch.Tensor,
#     target: torch.Tensor,
#     mask: Optional[torch.Tensor] = None,
#     sample_weight: Optional[torch.Tensor] = None,
#     target_weight: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     pred = _as_2d(pred)
#     target = _as_2d(target)
#     losses = [weighted_pearson_ic_loss(pred[:, h], target[:, h], mask, sample_weight) for h in range(pred.size(1))]
#     loss = torch.stack(losses)
#     if target_weight is not None:
#         w = target_weight.to(device=loss.device, dtype=loss.dtype)
#         w = w / w.sum().clamp_min(1e-8)
#         return (loss * w).sum()
#     return loss.mean()


# def pairwise_rank_loss(
#     pred: torch.Tensor,
#     target: torch.Tensor,
#     mask: Optional[torch.Tensor] = None,
#     max_pairs: int = 20000,
#     margin_scale: float = 1.0,
# ) -> torch.Tensor:
#     if pred.dim() > 1:
#         pred = pred[:, 0]
#     if target.dim() > 1:
#         target = target[:, 0]
#     pred = pred.view(-1)
#     target = target.view(-1)
#     valid = torch.isfinite(pred) & torch.isfinite(target)
#     if mask is not None:
#         valid = valid & mask.bool().view(-1)
#     pred = pred[valid]
#     target = target[valid]
#     n = pred.numel()
#     if n <= 2:
#         return pred.new_tensor(0.0)
#     num_pairs = min(max_pairs, n * (n - 1) // 2)
#     i = torch.randint(0, n, (num_pairs,), device=pred.device)
#     j = torch.randint(0, n, (num_pairs,), device=pred.device)
#     diff_y = target[i] - target[j]
#     non_tie = diff_y.abs() > 1e-12
#     if non_tie.sum() == 0:
#         return pred.new_tensor(0.0)
#     i = i[non_tie]
#     j = j[non_tie]
#     sign = torch.sign(diff_y[non_tie])
#     score = sign * (pred[i] - pred[j])
#     return F.softplus(-margin_scale * score).mean()


# def huber_regression_loss(
#     pred: torch.Tensor,
#     target: torch.Tensor,
#     mask: Optional[torch.Tensor] = None,
#     sample_weight: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     pred = _as_2d(pred)
#     target = _as_2d(target)
#     valid = torch.isfinite(pred).all(dim=1) & torch.isfinite(target).all(dim=1)
#     if mask is not None:
#         valid = valid & mask.bool().view(-1)
#     pred = pred[valid]
#     target = target[valid]
#     if pred.numel() == 0:
#         return pred.new_tensor(0.0)
#     loss = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=1)
#     if sample_weight is not None:
#         w = sample_weight.view(-1)[valid].to(device=loss.device, dtype=loss.dtype).clamp_min(0)
#         w = w / w.sum().clamp_min(1e-8)
#         return (loss * w).sum()
#     return loss.mean()


# def direction_loss(logit: torch.Tensor, target_return: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#     if logit.dim() > 1:
#         logit = logit[:, 0]
#     if target_return.dim() > 1:
#         target_return = target_return[:, 0]
#     logit = logit.view(-1)
#     label = (target_return.view(-1) > 0).float()
#     valid = torch.isfinite(logit) & torch.isfinite(target_return.view(-1))
#     if mask is not None:
#         valid = valid & mask.bool().view(-1)
#     logit = logit[valid]
#     label = label[valid]
#     if logit.numel() == 0:
#         return logit.new_tensor(0.0)
#     return F.binary_cross_entropy_with_logits(logit, label)




class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"sequence length {x.size(1)} exceeds max_len {self.pe.size(1)}")
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


class EventBroadcastFusion(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # 保存配置参数
        self.hidden_dim = cfg.hidden_dim          # 隐藏层维度 H
        self.max_effect_days = cfg.max_effect_days # 事件最大有效影响天数（如 30天）

        # ------------------------------------------------------------
        # 1. 事件编码器（浅层特征提取）
        # 输入: [B, K, D_event] -> 输出: [B, K, H]
        # 作用：将原始事件特征向量映射到与市场序列相同的隐藏空间
        # ------------------------------------------------------------
        self.event_proj = nn.Sequential(
            nn.LayerNorm(cfg.event_dim),          # 对原始特征做 LayerNorm，稳定训练
            nn.Linear(cfg.event_dim, cfg.hidden_dim), # 线性映射 D_event -> H
            nn.GELU(),                            # 非线性激活
            nn.Dropout(cfg.dropout),              # 随机失活防过拟合
        )

        # ------------------------------------------------------------
        # 2. 可选的事件间自注意力编码器（深层特征提取）
        # 输入: [B, K, H] -> 输出: [B, K, H]
        # 作用：让不同公告之间互相交换信息（如"分红"与"诉讼"的关联）
        # ------------------------------------------------------------
        if cfg.event_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.hidden_dim,           # H
                nhead=cfg.num_heads,              # 多头注意力头数
                dim_feedforward=cfg.hidden_dim * 4, # FFN 中间层维度 4H
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,                 # 输入形状为 [B, K, H]
                norm_first=True,                  # Pre-LN（先归一化再计算），训练更稳
            )
            self.event_encoder = nn.TransformerEncoder(layer, num_layers=cfg.event_layers)
        else:
            self.event_encoder = None             # 若层数为0，则完全跳过自注意力

        self.event_norm = nn.LayerNorm(cfg.hidden_dim) # 事件编码后的最终归一化层

        # ------------------------------------------------------------
        # 3. 交叉注意力（Cross-Attention）的 Q/K/V 投影
        # 注意：Q 来自市场序列 [B, L, H]，K/V 来自事件 [B, K, H]
        # ------------------------------------------------------------
        self.q_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim) # Market -> Query
        self.k_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim) # Event -> Key
        self.v_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim) # Event -> Value

        # ------------------------------------------------------------
        # 4. 时间衰减参数（可学习标量）
        # 控制事件影响力随年龄增长的衰减速度，初始值 1.0
        # 经过 softplus 保证恒为正数
        # ------------------------------------------------------------
        self.decay = nn.Parameter(torch.tensor(1.0))

        # ------------------------------------------------------------
        # 5. Delta 投影层：根据融合后的上下文生成“残差增量”
        # 输入: context [B, L, H] -> 输出: delta [B, L, H]
        # 最终输出 = market_seq + gate * delta（残差结构）
        # ------------------------------------------------------------
        self.delta_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), # 最后一层
        )

        # ------------------------------------------------------------
        # 6. 门控机制（Gating Mechanism）
        # 决定“在多大程度上”将事件信息注入市场序列
        # 输入特征维度：market_seq(H) + context(H) + diff(H) + has_event(1) + age_feature(1)
        # 总维度 = H*3 + 2
        # ------------------------------------------------------------
        self.gate = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 3 + 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Sigmoid(),                         # 输出 0~1 之间的门控值
        )

        self.out_norm = nn.LayerNorm(cfg.hidden_dim) # 最终输出的归一化

        # ------------------------------------------------------------
        # 7. 关键初始化：Delta 最后一层权重和偏置置零
        # 使模型初始状态为 fused_seq = market_seq（零增量）
        # 防止随机初始化的事件分支在训练初期破坏预训练好的市场主干
        # ------------------------------------------------------------
        nn.init.zeros_(self.delta_proj[-1].weight)
        nn.init.zeros_(self.delta_proj[-1].bias)

    def encode_events(self, event_x: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
        # event_x: [B, K, D_event]  原始事件特征
        # event_mask: [B, K]  布尔张量，True=该位置有真实事件，False=填充(PAD)

        # ------------------------------------------------------------
        # Step 1: 投影到隐藏空间 [B, K, D_event] -> [B, K, H]
        # ------------------------------------------------------------
        event_h = self.event_proj(event_x)

        # ------------------------------------------------------------
        # Step 2: 如果配置了 0 层 Transformer（轻量模式）
        # 直接归一化并乘上 mask，将填充位置清零
        # ------------------------------------------------------------
        if self.event_encoder is None:
            return self.event_norm(event_h) * event_mask.unsqueeze(-1).to(event_h.dtype)
            # event_mask.unsqueeze(-1) -> [B, K, 1] 广播相乘，填充位变0

        # ------------------------------------------------------------
        # Step 3: 准备 Transformer 所需的 key_padding_mask
        # PyTorch 要求: True 表示忽略该位置
        # 外部 event_mask: True=有效 -> 取反后变为 False=有效
        # ------------------------------------------------------------
        key_padding_mask = ~event_mask.bool()  # [B, K]，True=填充要忽略

        # ------------------------------------------------------------
        # Step 4: 处理极端情况——某只股票完全没有事件（整个 K 维度全为 False）
        # 如果不处理，Transformer 会因序列长度为0而报错
        # 这里通过“造假”强行保留第0个位置作为占位符
        # ------------------------------------------------------------
        empty = event_mask.sum(dim=1) == 0      # [B]，标记哪些样本全空
        if empty.any():
            # 克隆避免影响原张量
            key_padding_mask = key_padding_mask.clone()
            # 将全空样本的第0个位置设为 False（表示该位置“有效”），骗过 Transformer
            key_padding_mask[empty, 0] = False
            # 同时将 event_h 的第0个位置置为全零向量
            event_h = event_h.clone()
            event_h[empty, 0, :] = 0.0          # 虽然是“有效”占位，但内容是0，不引入噪音

        # ------------------------------------------------------------
        # Step 5: 执行事件间自注意力编码 [B, K, H] -> [B, K, H]
        # ------------------------------------------------------------
        event_h = self.event_encoder(event_h, src_key_padding_mask=key_padding_mask)

        # ------------------------------------------------------------
        # Step 6: 归一化并再次乘以 mask，将填充位（及全空样本的占位符）重新清零
        # 保证对外输出时，所有无效位置严格为0
        # ------------------------------------------------------------
        event_h = self.event_norm(event_h)
        event_h = event_h * event_mask.unsqueeze(-1).to(event_h.dtype)
        return event_h  # [B, K, H]

    def forward(
        self,
        market_seq: torch.Tensor,   # [B, L, H] 市场编码器输出的序列
        event_x: torch.Tensor,      # [B, K, D_event] 原始事件特征
        event_mask: torch.Tensor,   # [B, K] True=有效事件
        event_age: torch.Tensor,    # [B, L, K] 每个市场时间步下，每个事件的“年龄”（距该天的天数）
    ) -> Tuple[...]:
        
        # ----------------------------------------------
        # 1. 维度合法性校验（略，但关键参数解释）
        # event_age 形状为 [B, L, K] 是关键：
        #   对于第 i 个股票，第 t 天，第 k 个公告，其年龄 = 今天 - 公告日
        # ----------------------------------------------
        # 校验代码略...

        # ----------------------------------------------
        # 2. 编码事件：获取事件语义向量 [B, K, H]
        # ----------------------------------------------
        event_h = self.encode_events(event_x, event_mask)  # [B, K, H]

        # ----------------------------------------------
        # 3. 计算 Cross-Attention 的 Q, K, V
        #    Q: 来自市场每一天 [B, L, H]
        #    K, V: 来自事件集合 [B, K, H]
        # ----------------------------------------------
        q = self.q_proj(market_seq)   # [B, L, H]
        k = self.k_proj(event_h)      # [B, K, H]
        v = self.v_proj(event_h)      # [B, K, H]

        # ----------------------------------------------
        # 4. 计算原始注意力分数（缩放点积）
        #    torch.einsum("blh,bkh->blk", q, k) 对最后两维做内积
        #    结果 [B, L, K]，表示每一天对所有事件的相似度
        # ----------------------------------------------
        score = torch.einsum("blh,bkh->blk", q, k) / math.sqrt(self.hidden_dim)  # [B, L, K]

        # ----------------------------------------------
        # 5. 构建有效性掩码 valid [B, L, K]
        #    必须同时满足三个条件才允许当前天关注该事件：
        #    1) 该事件本身是有效的 (event_mask == 1)
        #    2) 事件年龄 >= 0（尚未发生的事件 age=-1 会被排除，防止未来信息泄露）
        #    3) 事件年龄 <= max_effect_days（过期的事件不再产生影响）
        # ----------------------------------------------
        valid = (
            event_mask.bool().unsqueeze(1)          # [B, 1, K] 广播到 [B, L, K]
            & (event_age >= 0)                     # [B, L, K]
            & (event_age <= self.max_effect_days)  # [B, L, K]
        )  # 最终 [B, L, K]

        # ----------------------------------------------
        # 6. 计算时间衰减偏置（Time Bias）
        #    age 越大，time_bias 越负，导致该事件得分越低
        #    self.decay 控制衰减速度（可学习）
        # ----------------------------------------------
        age = event_age.clamp_min(0).to(dtype=market_seq.dtype)  # [B, L, K]，将负数截断为0
        # softplus(decay) 保证衰减系数恒为正
        # log1p(age) 使年龄增长带来的衰减逐渐放缓
        # 除以 log1p(max_effect_days+1) 做归一化，使 bias 在合理范围内
        time_bias = -F.softplus(self.decay) * torch.log1p(age) / math.log1p(self.max_effect_days + 1.0)
        # time_bias 形状 [B, L, K]

        # 将时间偏置加到原始分数上
        score = score + time_bias  # [B, L, K]

        # ----------------------------------------------
        # 7. 掩码填充：将无效位置（valid=False）的分数设为 -1e9（近似负无穷）
        #    经过 softmax 后，这些位置的注意力权重将变为 0
        # ----------------------------------------------
        score = score.masked_fill(~valid, -1e9)  # [B, L, K]

        # ----------------------------------------------
        # 8. 计算注意力权重 Softmax
        #    对每个时间步（L 维度）在事件维度（K）上做归一化
        # ----------------------------------------------
        attn = torch.softmax(score, dim=-1)  # [B, L, K]

        # ----------------------------------------------
        # 9. 安全重归一化（防止某天没有任何有效事件导致 Softmax 输出 NaN）
        #    将无效位置的权重强制置0，再除以总和（clamp 防除0）
        # ----------------------------------------------
        attn = attn * valid.to(dtype=attn.dtype)          # 无效位置权重清零
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)  # 重新归一化

        # ----------------------------------------------
        # 10. 加权求和得到上下文向量 Context [B, L, H]
        #     即：每一天，根据注意力权重，对所有事件向量 V 做加权平均
        # ----------------------------------------------
        context = torch.einsum("blk,bkh->blh", attn, v)  # [B, L, H]

        # ----------------------------------------------
        # 11. has_event [B, L, 1]
        #      标记该天是否存在任何有效事件（用于后续门控）
        # ----------------------------------------------
        has_event = valid.any(dim=-1, keepdim=True).to(dtype=market_seq.dtype)  # [B, L, 1]

        # ----------------------------------------------
        # 12. 计算“最近有效事件的最小年龄”作为额外特征 age_feature
        #      对没有事件的 day，该特征为 0
        # ----------------------------------------------
        min_age = age.masked_fill(~valid, 1e9).min(dim=-1, keepdim=True).values  # [B, L, 1]
        # 如果该天没有事件，has_event=False，则 min_age 被置为 0
        min_age = torch.where(has_event.bool(), min_age, torch.zeros_like(min_age))
        # 归一化到 [0, 1] 区间
        age_feature = torch.log1p(min_age) / math.log1p(self.max_effect_days + 1.0)  # [B, L, 1]

        # ----------------------------------------------
        # 13. 组装门控输入（Gate Input）
        #      包含：原始市场特征、上下文、差值、是否有事件、最小年龄
        # ----------------------------------------------
        gate_input = torch.cat(
            [
                market_seq,            # [B, L, H]
                context,               # [B, L, H]
                market_seq - context,  # [B, L, H] 差值特征
                has_event,             # [B, L, 1]
                age_feature            # [B, L, 1]
            ],
            dim=-1
        )  # 总维度 H*3 + 2

        # ----------------------------------------------
        # 14. 计算门控值 Gate [B, L, H]
        #      值域 (0,1)，控制事件信息的注入强度
        #      如果 has_event=0，则 gate 强行归零（乘以 has_event）
        # ----------------------------------------------
        gate = self.gate(gate_input) * has_event  # [B, L, H]

        # ----------------------------------------------
        # 15. 计算残差增量 Delta [B, L, H]
        #      从 context 中提取要修改的方向
        # ----------------------------------------------
        delta = self.delta_proj(context)  # [B, L, H]

        # ----------------------------------------------
        # 16. 最终融合输出（残差连接）
        #      fused_seq = market_seq + gate * delta
        #      当训练初期 delta 为零时，fused_seq 完全等于 market_seq
        # ----------------------------------------------
        fused_seq = self.out_norm(market_seq + gate * delta)  # [B, L, H]

        # ----------------------------------------------
        # 17. 返回结果（含中间变量用于调试/可视化）
        # ----------------------------------------------
        return fused_seq, context, attn, gate, has_event.squeeze(-1)
        # fused_seq: [B, L, H] 融合后的市场序列
        # context:   [B, L, H] 从事件中提取的上下文
        # attn:      [B, L, K] 注意力权重（可解释性分析）
        # gate:      [B, L, H] 门控值
        # has_event: [B, L]   每日是否有有效事件


class PITEventResidualAlphaModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.market_encoder = MarketEncoder(cfg)
        self.event_fusion = EventBroadcastFusion(cfg)

        self.base_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_targets),
        )
        self.event_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_targets),
        )
        self.event_scale = nn.Parameter(torch.tensor(float(cfg.event_scale_init)))

        self.cls_head = None
        if cfg.output_cls_logit:
            self.cls_head = nn.Sequential(
                nn.LayerNorm(cfg.hidden_dim),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim // 2, cfg.num_targets),
            )

    def forward(
        self,
        market_x: torch.Tensor,
        event_x: torch.Tensor,
        event_mask: torch.Tensor,
        event_age: torch.Tensor,
        market_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        market_seq, market_vec = self.market_encoder(market_x, market_mask)
        base_pred = self.base_head(market_vec)

        fused_seq, event_context_seq, event_attn, event_gate, has_event_seq = self.event_fusion(
            market_seq=market_seq,
            event_x=event_x,
            event_mask=event_mask,
            event_age=event_age,
        )

        fused_vec = fused_seq[:, -1] if market_mask is None else masked_mean(fused_seq, market_mask, dim=1)
        event_residual = self.event_head(fused_vec)
        pred = base_pred + self.event_scale * event_residual

        out: Dict[str, torch.Tensor] = {
            "pred": pred,
            "base_pred": base_pred,
            "event_residual": event_residual,
            "fused_vec": fused_vec,
            "market_vec": market_vec,
            "event_context_seq": event_context_seq,
            "event_attention": event_attn,
            "event_gate": event_gate,
            "has_event_seq": has_event_seq,
            "has_recent_event": has_event_seq.any(dim=1),
            "event_scale": self.event_scale.detach(),
        }
        if self.cls_head is not None:
            out["cls_logit"] = self.cls_head(fused_vec)

        # Keep single-target output compatible with old code.
        if pred.size(-1) == 1:
            for key in ["pred", "base_pred", "event_residual"]:
                out[key] = out[key].squeeze(-1)
            if "cls_logit" in out:
                out["cls_logit"] = out["cls_logit"].squeeze(-1)
        return out


# def compute_loss(
#     outputs: Dict[str, torch.Tensor],
#     y: torch.Tensor,
#     stock_mask: Optional[torch.Tensor] = None,
#     sample_weight: Optional[torch.Tensor] = None,
#     target_weight: Optional[torch.Tensor] = None,
#     w_ic: float = 0.60,
#     w_pair: float = 0.25,
#     w_reg: float = 0.10,
#     w_cls: float = 0.00,
#     w_event_aux: float = 0.05,
# ) -> Dict[str, torch.Tensor]:
#     """
#     Main loss: cross-sectional ranking/regression on final pred.

#     Auxiliary event loss: only on samples with recent PIT-visible reports, train
#     the event branch to explain y - base_pred.detach(). This matches the idea
#     that financial reports should add incremental information over pure market
#     Transformer features.
#     """
#     pred = outputs["pred"]
#     loss_ic = multi_target_ic_loss(pred, y, stock_mask, sample_weight, target_weight)
#     loss_pair = pairwise_rank_loss(pred, y, stock_mask)
#     loss_reg = huber_regression_loss(pred, y, stock_mask, sample_weight)
#     total = w_ic * loss_ic + w_pair * loss_pair + w_reg * loss_reg

#     loss_cls = pred.new_tensor(0.0)
#     if "cls_logit" in outputs and w_cls > 0:
#         loss_cls = direction_loss(outputs["cls_logit"], y, stock_mask)
#         total = total + w_cls * loss_cls

#     loss_event_aux = pred.new_tensor(0.0)
#     if w_event_aux > 0 and "event_residual" in outputs and "base_pred" in outputs:
#         event_mask = outputs.get("has_recent_event", None)
#         if event_mask is not None:
#             aux_mask = event_mask.bool()
#             if stock_mask is not None:
#                 aux_mask = aux_mask & stock_mask.bool().view(-1)
#             if aux_mask.sum() > 2:
#                 residual_target = _as_2d(y) - _as_2d(outputs["base_pred"]).detach()
#                 loss_event_aux = huber_regression_loss(outputs["event_residual"], residual_target, aux_mask)
#                 total = total + w_event_aux * loss_event_aux

#     return {
#         "loss": total,
#         "loss_ic": loss_ic.detach(),
#         "loss_pair": loss_pair.detach(),
#         "loss_reg": loss_reg.detach(),
#         "loss_cls": loss_cls.detach(),
#         "loss_event_aux": loss_event_aux.detach(),
#     }


def make_event_sample_weight(event_age: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
    """
    Optional sample weights for training. event_age is [B, L, K].
    The last market day is the prediction day.
    """
    last_age = event_age[:, -1, :]
    valid = event_mask.bool() & (last_age >= 0)
    recent_20 = valid & (last_age <= 20)
    recent_60 = valid & (last_age > 20) & (last_age <= 60)
    recent_120 = valid & (last_age > 60) & (last_age <= 120)
    w = torch.ones(event_age.size(0), device=event_age.device, dtype=torch.float32)
    w = w + 2.0 * recent_20.any(dim=1).float()
    w = w + 1.0 * recent_60.any(dim=1).float()
    w = w + 0.5 * recent_120.any(dim=1).float()
    return w


def make_toy_batch(B: int = 8, L: int = 60, Dm: int = 12, K: int = 4, De: int = 60, seed: int = 7) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    market_x = torch.randn(B, L, Dm, generator=g) * 0.05
    event_x = torch.randn(B, K, De, generator=g) * 0.10
    event_mask = torch.zeros(B, K, dtype=torch.long)
    event_age = torch.full((B, L, K), -1, dtype=torch.long)
    for b in range(B):
        c = int(torch.randint(0, K + 1, (1,), generator=g))
        event_mask[b, :c] = 1
        for k in range(c):
            effective_pos = int(torch.randint(0, L, (1,), generator=g))
            event_age[b, :, k] = torch.arange(L) - effective_pos
            # quarter one-hot example
            event_x[b, k, k % 4] = 1.0
    event_x = event_x * event_mask.unsqueeze(-1)
    y = torch.randn(B, generator=g) * 0.02
    stock_mask = torch.ones(B, dtype=torch.long)
    return {
        "market_x": market_x,
        "event_x": event_x,
        "event_mask": event_mask,
        "event_age": event_age,
        "stock_mask": stock_mask,
        "y": y,
    }


def demo_run() -> None:
    batch = make_toy_batch()
    cfg = ModelConfig(
        market_dim=batch["market_x"].shape[-1],
        event_dim=batch["event_x"].shape[-1],
        hidden_dim=64,
        num_heads=4,
        market_layers=2,
        event_layers=1,
        num_targets=1,
        max_effect_days=120,
    )
    model = PITEventResidualAlphaModel(cfg)
    out = model(batch["market_x"], batch["event_x"], batch["event_mask"], batch["event_age"])
    sample_weight = make_event_sample_weight(batch["event_age"], batch["event_mask"])
    loss = compute_loss(out, batch["y"], batch["stock_mask"], sample_weight=sample_weight)
    print("pred", tuple(out["pred"].shape))
    print("event_attention", tuple(out["event_attention"].shape))
    print({k: float(v) for k, v in loss.items()})


if __name__ == "__main__":
    demo_run()
