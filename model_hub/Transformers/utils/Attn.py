import torch
import torch.nn as nn
from math import sqrt
from .masking import TriangularCausalMask




class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask=attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn




class FullAttention(nn.Module):
    """标准全注意力。attn_mask 中 True 表示屏蔽。"""

    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, _ = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag and attn_mask is None:
            attn_mask = TriangularCausalMask(B, L, device=queries.device).mask
        if attn_mask is not None:
            if hasattr(attn_mask, "mask"):
                attn_mask = attn_mask.mask
            attn_mask = attn_mask.to(device=queries.device, dtype=torch.bool)
            scores = scores.masked_fill(attn_mask, torch.finfo(scores.dtype).min)

        attention = torch.softmax(scale * scores, dim=-1)
        if attn_mask is not None:
            # 全部位置被屏蔽时，将注意力权重保持为零。
            attention = attention.masked_fill(attn_mask, 0.0)
            attention = attention / attention.sum(
                dim=-1, keepdim=True
            ).clamp_min(torch.finfo(attention.dtype).eps)

        attention = self.dropout(attention)
        output = torch.einsum("bhls,bshd->blhd", attention, values)

        if self.output_attention:
            return output.contiguous(), attention
        return output.contiguous(), None
