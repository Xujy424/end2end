from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.args import BaseArg



def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    return (x * mask).sum(dim=dim) / mask.sum(dim=dim).clamp_min(eps)


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
    def __init__(self, cfg: ERED_Arg):
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
    def __init__(self, cfg: ERED_Arg):
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
    ):
        
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


class ERED_Model(nn.Module):
    def __init__(self, cfg: ERED_Arg):
        super().__init__()
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

        # self.cls_head = None
        # if cfg.output_cls_logit:
        #     self.cls_head = nn.Sequential(
        #         nn.LayerNorm(cfg.hidden_dim),
        #         nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
        #         nn.GELU(),
        #         nn.Dropout(cfg.dropout),
        #         nn.Linear(cfg.hidden_dim // 2, cfg.num_targets),
        #     )

    def forward(self, x) -> Dict[str, torch.Tensor]:
        market_x, event_x, event_mask, event_age, market_mask = x['dailyset'], x['eventvec'], x['eventmask'], x['eventage'], None

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

        return pred


class ERED_Arg(BaseArg):
    d_fields = [
        'close_zscore','open_zscore','high_zscore','low_zscore','logvolume_zscore','turnover_zscore',
        'close_pct','open_pct','high_pct','low_pct','logvolume_pct','turnover_pct',
        'close2open','high2open','low2open','high2low','high2close','low2close',
    ]
    m_fields = ['close2dopen','high2dopen','low2dopen','ppos','volume_adj2rollmean','amount2rollmean']
    
    # event_fields = [
    #     'basic_eps_yoy',
    #     'total_operating_revenue_yoy',
    #     'np_parent_company_owners_yoy',
    #     'net_operate_cash_flow_yoy',
    #     'roe_yoy','l2a_yoy','gpm_yoy','npm_yoy',
    #     # 'basic_eps_qoq',
    #     # 'total_operating_revenue_qoq',
    #     # 'np_parent_company_owners_qoq',
    #     # 'net_operate_cash_flow_qoq',
    #     # 'roe_qoq','l2a_qoq','gpm_qoq','npm_qoq',
    #     'eps_ue_sue', 'or_ue_sue', 'np_ue_sue', 'roe_ue_sue',
    #     #'log_distance'
    #     'pct_20rollstd','pct_20rollmean','turnover_20rollstd','turnover_20rollmean'
    # ]
    event_fields=[
        'ROETTM', 'ROICTTM', 'GrossIncomeRatioTTM', 'NetProfitRatioTTM',
        'PeriodCostsRateTTM', 'AdminiExpenseRateTTM',
        'TotalAssetTRateTTM', 'ARTRate', 'InventoryTRate',
        'DebtAssetsRatio', 'LongDebtRatio', 
        'NPParentCompanyCutYOY', 'TotalAssetGrowRate', 'NetOperateCashFlowYOY',
        'NOCFToOperatingNITTM', 'SaleServiceCashToORTTM', 'OperCashInToAsset',
        'FixAssetRatio', 'IntangibleAssetRatio',
        #'DividendPaidRatio', 'RetainedEarningRatio'
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
                            "label": "Y.10D",
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
                            # 'timecode': {
                            #     'freq': 'daily', # 'intraday'
                            #     'lag': 20,
                            # },
                            "eventvec":{
                                "data_path": "/data/xujiayi/xjy/research_factors/model_input/ered_v2/",
                                "lag":20,
                                "max_events":8
                            },
                            "eventmask":{
                                "data_path": "/data/xujiayi/xjy/research_factors/model_input/ered_v2/",
                                "lag":20,
                                "max_events":8
                            },
                            "eventage":{
                                "data_path": "/data/xujiayi/xjy/research_factors/model_input/ered_v2/",
                                "lag":20,
                                "max_events":8
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
                    "cfg":{
                        "num_targets": 1,
                        "market_dim": len(self.d_fields),
                        "event_dim": len(self.event_fields),
                        "hidden_dim": 64,
                        "market_layers": 2,
                        "event_layers": 2,
                        "num_heads": 4,
                        "dropout": 0.5,
                        "max_effect_days": 30,
                        "event_scale_init": 0.1,
                        "max_len": 60,
                    }
                    
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