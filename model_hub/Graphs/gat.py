import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Tuple



import torch.nn.functional as F

def scatter_softmax(src: torch.Tensor, index: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    沿着 dim 维度, 对每个 index 组内做 softmax。
    等价于 torch_scatter.scatter_softmax, 但纯 PyTorch 实现。
    """
    max_val = torch.zeros_like(src).scatter_reduce(
        dim, index.unsqueeze(-1).expand_as(src), src,
        reduce='amax', include_self=False
    )
    src = src - max_val[index]                     # 数值稳定
    exp_src = src.exp()
    sum_exp = torch.zeros_like(exp_src).scatter_reduce(
        dim, index.unsqueeze(-1).expand_as(exp_src), exp_src,
        reduce='sum', include_self=False
    )
    return exp_src / (sum_exp[index] + 1e-12)

class GATHead(nn.Module):
    """
        单头注意力（GAT 原文）
    输入：
        h: [N, in_features]        节点特征
        edge_index: [2, E]         边索引
    输出：
        [N, out_features]
    """
    def __init__(self, in_features: int, out_features: int, alpha=0.2, dropout=0.0):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = h.size(0)
        i, j = edge_index
        # 线性变换 Wh
        Wh = self.W(h)
        # 注意力系数 e_ij
        e = self.leakyrelu(self.a(torch.cat([Wh[i], Wh[j]], dim=-1)))
        # softmax 归一化
        # alpha = torch.full((N, N), torch.finfo(Wh.dtype).min, device=h.device, dtype=Wh.dtype)
        # alpha[i, j] = e.squeeze(-1)
        # alpha = F.softmax(alpha, dim=1)[i, j].unsqueeze(-1)
        alpha = scatter_softmax(e, i, dim=0)
        alpha = self.dropout(alpha)
        # 加权聚合
        out = torch.zeros_like(Wh)
        source = (alpha * Wh[j]).to(out.dtype)
        out.index_add_(0, i, source)
        return out


class GAT(nn.Module):
    """
       完整 GAT（论文原版）
   输入：[N, in_dim], [2, E]
   输出：[N, out_dim]
   """
    def __init__(self, in_dim, hidden_dim, out_dim, heads=8, alpha=0.2, dropout=0.6):
        super().__init__()
        self.dropout = dropout
        # 多头注意力（原文）
        self.attentions = nn.ModuleList([
            GATHead(in_dim, hidden_dim, alpha, dropout) for _ in range(heads)
        ])
        # 输出头
        self.out_att = GATHead(hidden_dim * heads, out_dim, alpha, dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)
        x = F.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, edge_index)
        return x

    # def build_graph_t(self, edge_seq, corr_threshold=0.6):
    #     """
    #     单日期图构建（极速向量化）
    #     edge_seq: [N, T, F]
    #     return: edge_index [2, E]
    #     """
    #     N, T, F = edge_seq.shape
    #
    #     # 所有股票对相关系数
    #     x = edge_seq.permute(0, 2, 1)  # [N, F, W]
    #     corr = torch.corrcoef(x.flatten(0, 1))[:N, :N]
    #     corr = corr.masked_fill(torch.isnan(corr), 0.0)
    #
    #     # 建边
    #     adj = (corr > corr_threshold).bool()
    #     adj.fill_diagonal_(True)
    #
    #     # 转 COO 边索引 [2, E]
    #     edges = adj.nonzero(as_tuple=True)
    #     edge_index = torch.stack(edges, dim=0)
    #     return edge_index