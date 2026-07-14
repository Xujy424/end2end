import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from utils import DataEmbedding, Inception_Block_V1



def FFT_for_Period(x, k=2):
    # x: [B, T, C] 输入：批次、序列长度、特征维度
    # 对 时序维度dim=1 做实数快速傅里叶变换，提取频率信息
    xf = torch.fft.rfft(x, dim=1)  # [B, T//2+1, C] 实FFT只保留正频率

    # 按振幅找周期：频率振幅越大，周期性越强
    # 先对B维度求平均 → 再对C特征维度求平均 → 得到每个频率的振幅
    frequency_list = abs(xf).mean(0).mean(-1)  # [T//2+1]
    
    frequency_list[0] = 0  # 直流分量（频率0）置0，不参与周期计算
    
    # 取振幅最大的前k个频率索引
    _, top_list = torch.topk(frequency_list, k)  # [k]
    
    # 转到numpy计算周期
    top_list = top_list.detach().cpu().numpy()
    
    # 周期 = 序列长度 // 对应频率（时序核心：频率↔周期换算）
    period = x.shape[1] // top_list  # [k] 输出k个周期长度
    
    # 返回：k个周期值 + 每个样本在topk频率上的振幅（后续做注意力权重）
    return period, abs(xf).mean(-1)[:, top_list]  # [k], [B, k]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len      # 输入序列长度
        self.pred_len = configs.pred_len    # 预测序列长度
        self.k = configs.top_k              # 提取k个主要周期
        
        # 卷积层：Inception多尺度卷积 + GELU激活
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()  # x: [B, T, N] 输入：批次、序列长度、隐层维度
        
        # 1. FFT自动提取数据的k个主要周期 + 周期振幅权重
        period_list, period_weight = FFT_for_Period(x, self.k)
        # period_list: [k]  k个周期长度；period_weight: [B, k]  周期重要性权重

        res = []
        # 2. 逐周期处理：把一维时序 → 二维周期图
        for i in range(self.k):
            period = period_list[i]  # 取出第i个周期长度
            
            # 填充：让总长度能被周期整除，方便reshape
            total_len = self.seq_len + self.pred_len  # 输入+预测总长度
            if total_len % period != 0:
                # 计算需要填充到的最小整倍数长度
                length = ((total_len // period) + 1) * period
                # 构造填充0张量：[B, 填充长度, N]
                padding = torch.zeros([B, length - total_len, N]).to(x.device)
                out = torch.cat([x, padding], dim=1)  # [B, length, N] 拼接填充
            else:
                length = total_len
                out = x  # [B, length, N] 无需填充
            
            # 3. 核心变形：1维时序 → 2维周期特征图
            # reshape: [B, 周期数, 周期长度, N]
            out = out.reshape(B, length // period, period, N)
            # 维度置换 → [B, N, 周期数, 周期长度] 适配卷积输入格式(通道在前)
            out = out.permute(0, 3, 1, 2).contiguous()
            
            # 4. 多尺度卷积提取2维周期图的特征
            out = self.conv(out)  # [B, N, 周期数, 周期长度] 特征提取
            
            # 5. 逆变形：2维特征图 → 恢复1维时序
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)  # [B, length, N]
            
            # 截断回原长度，加入结果列表
            res.append(out[:, :total_len, :])  # [B, seq_len+pred_len, N]
        
        # 堆叠k个周期分支的结果
        res = torch.stack(res, dim=-1)  # [B, T, N, k]

        # 6. 自适应融合：用FFT得到的周期权重做软加权
        period_weight = F.softmax(period_weight, dim=1)  # [B, k] 权重归一化
        # 维度扩展，适配res形状：[B, T, N, k]
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        
        # 按权重求和融合k个周期分支
        res = torch.sum(res * period_weight, -1)  # [B, T, N]

        # 7. 残差连接：缓解深度网络梯度消失
        res = res + x  # [B, T, N]
        
        return res



class Model(nn.Module):
    """
    TimesNet 完整模型：支持预测/补全/异常检测/分类
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name        # 任务类型
        self.seq_len = configs.seq_len            # 输入长度
        self.label_len = configs.label_len        # 标签长度
        self.pred_len = configs.pred_len          # 预测长度
        
        # 堆叠多层 TimesBlock 核心模块
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        # 数据嵌入：把原始特征 → 模型隐层特征 + 时间编码
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.layer = configs.e_layers             # 编码器层数
        self.layer_norm = nn.LayerNorm(configs.d_model)  # 层归一化

        # 不同任务的输出头
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # 时序对齐：输入长度 → 输入+预测长度
            self.predict_linear = nn.Linear(self.seq_len, self.seq_len + self.pred_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)  # 输出投影
            
        if self.task_name in ['imputation', 'anomaly_detection']:
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
            
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    # ------------------- 核心任务：长/短时序预测 -------------------
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, T, C] 原始输入序列；x_mark_enc: [B, T, D] 时间戳编码
        
        # 1. 归一化（来自Non-stationary Transformer，稳定非平稳时序训练）
        means = x_enc.mean(1, keepdim=True).detach()  # [B,1,C] 按序列求均值
        x_enc = x_enc - means                         # 去均值
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [B,1,C] 求标准差
        x_enc = x_enc / stdev                         # 标准化

        # 2. 嵌入层：原始特征 → 模型隐层特征 [B,T,C] → [B,T,d_model]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # 3. 线性对齐：把输入序列长度 → 输入+预测总长度 [B,T,d_model] → [B,T+L,d_model]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 4. 堆叠 TimesBlock 提取周期特征
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))  # [B, T+L, d_model]
        
        # 5. 输出投影：隐层特征 → 原始数据维度
        dec_out = self.projection(enc_out)  # [B, T+L, c_out]

        # 6. 反归一化：恢复数据真实尺度
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len + self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len + self.pred_len, 1)
        
        return dec_out

    # ------------------- 时序补全：缺失值填充 -------------------
    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # mask: [B,T,C] 掩码，1=有值，0=缺失
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)  # 只对有值求均值
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # 多层TimesBlock提取特征
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)

        # 反归一化
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len + self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len + self.pred_len, 1)
        return dec_out

    # ------------------- 异常检测 -------------------
    def anomaly_detection(self, x_enc):
        # 归一化
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # 多层TimesBlock
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)

        # 反归一化
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len + self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len + self.pred_len, 1)
        return dec_out

    # ------------------- 时序分类 -------------------
    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # 多层TimesBlock
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # 掩码填充部分
        output = output.reshape(output.shape[0], -1)  # [B, T*d_model] 展平
        output = self.projection(output)  # [B, num_class] 分类输出
        return output

    # ------------------- 前向传播总入口 -------------------
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # 只返回预测部分 [B, L, D]
        
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, T, D]
        
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, T, D]
        
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        
        return None

