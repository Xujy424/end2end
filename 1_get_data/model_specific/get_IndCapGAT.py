import talib
import numpy as np
import pandas as pd
import bottleneck as bn
from typing import Dict, Tuple
import os

from get_MultiscaleTransformer_data import VP_features, load_basic_feats
from utils import Calculator


def load_industry_mask(data_dir: str) -> Dict[str, np.ndarray]:
    file_path = os.path.join(data_dir, f"industry.bin")
    mask = np.memmap(file_path, shape=(3400,5422), dtype=float)  # 大文件用内存映射
    return mask

def load_fundamental_feats(data_dir: str) -> Dict[str, np.ndarray]:
    features = {}
    for feat_name in ['pe','pb','roe','debt_ratio','revenue_yoy','ocf']:
        file_path = os.path.join(data_dir, f"{feat_name}.bin")
        features[feat_name] = np.memmap(file_path, shape=(3400,5422), dtype=float)  # 大文件用内存映射
    return features


class GAT_features(VP_features, Calculator):
    """
    完全对齐西南证券研报：indcap-GAT模型特征计算
    包含：共享节点特征 + 资金流向图特征 + 行业关联图特征
    输入维度：(T=3400, N=5422) 与原有VP_features完全一致
    """

    def __init__(
            self,
            basic_features: Dict[str, np.ndarray],
            industry_mask: np.ndarray,  # (T,N) 申万一级行业编码（整数）
            fundamental_data: Dict[str, np.ndarray]  # 财务指标，均为(T,N)格式
    ):
        super().__init__(basic_features)
        self.industry = industry_mask  # 申万一级行业分类
        # 财务指标（日频，季度数据需提前向前填充）
        self.pe = fundamental_data['pe']
        self.pb = fundamental_data['pb']
        self.roe = fundamental_data['roe']
        self.debt_ratio = fundamental_data['debt_ratio']  # 资产负债率
        self.revenue_growth = fundamental_data['revenue_yoy']  # 营收同比增速
        self.ocf = fundamental_data['ocf']  # 经营活动现金流净额

    # ==============================================
    # 1. 共享节点特征（行业图+资金图通用）
    # ==============================================
    def calc_ret_n(self, window: int) -> np.ndarray:
        """过去n个交易日累计收益率 ret_n"""
        return self.close / np.roll(self.close, window, axis=0) - 1

    def calc_mom_m_n(self, m: int, n: int) -> np.ndarray:
        """收益率动量调整 mom_m_n = 过去m日累计 - 过去n日累计"""
        ret_m = self.calc_ret_n(m)
        ret_n = self.calc_ret_n(n)
        return ret_m - ret_n

    def calc_ret_std_n(self, window: int) -> np.ndarray:
        """过去n日收益率标准差 ret_std_n"""
        return self.rolling_nanstd(self.pct, window=window)

    def calc_turn_std_m(self, window: int) -> np.ndarray:
        """换手率波动比 = 过去n日换手率标准差 / 均值"""
        turn_std = self.rolling_nanstd(self.turnover, window=window)
        turn_mean = self.rolling_nanmean(self.turnover, window=window)
        res = self.safe_div(turn_std, turn_mean)
        return res

    def calc_turn_nm(self, window: int) -> np.ndarray:
        """过去n日换手率均值取对数"""
        turn_mean = self.rolling_nanmean(self.turnover, window=window)
        return np.log(np.where(turn_mean < 1e-8, np.nan, turn_mean))

    def calc_pospct_std(self, window: int) -> np.ndarray:
        """正低波因子：过去n日正收益率标准差"""
        pos_pct = np.where(self.pct > 0, self.pct, np.nan)
        return self.rolling_nanstd(pos_pct, window=window)

    def calc_negpct_std(self, window: int) -> np.ndarray:
        """负低波因子：过去n日负收益率标准差"""
        neg_pct = np.where(self.pct < 0, self.pct, np.nan)
        return self.rolling_nanstd(neg_pct, window=window)

    def calc_pct_vol_cor(self, window: int) -> np.ndarray:
        """量价支撑度：过去n日收益率与成交量的相关系数"""
        return self.rolling_nancorr(self.pct, self.volume, window=window)

    # ==============================================
    # 2. 资金流向图特征
    # ==============================================
    def calc_aavg(self, window: int = 20) -> np.ndarray:
        """近20日平均成交额"""
        return self.rolling_nanmean(self.amount, window=window)

    def calc_illiq(self, window: int = 20) -> np.ndarray:
        """Amihud非流动性指标：20日平均(|收益率|/成交额) 取对数"""
        daily_illiq = self.safe_div(np.abs(self.pct), self.amount)
        illiq_mean = self.rolling_nanmean(daily_illiq, window=window)
        return np.log(np.where(illiq_mean < 1e-12, np.nan, illiq_mean))

    def calc_stda(self, window: int = 20) -> np.ndarray:
        """成交额波动率：20日成交额标准差"""
        return self.rolling_nanstd(self.amount, window=window)

    def calc_f(self) -> np.ndarray:
        """资金流向指标：sign(涨跌幅)*成交额"""
        return np.sign(self.pct) * self.amount

    def calc_delta_tr(self, window: int = 5) -> np.ndarray:
        """换手率变化率：(当日换手率-5日前换手率)/5日前换手率"""
        tr_lag = np.roll(self.turnover, window, axis=0)
        tr_lag[:window] = np.nan
        res = self.safe_div(self.turnover - tr_lag, tr_lag)
        return res

    def calc_cor_pv(self, window: int = 20) -> np.ndarray:
        """价格-成交量相关系数：过去20日收盘价与成交量相关系数"""
        return self.rolling_nancorr(self.close, self.volume, window=window)

    # ==============================================
    # 3. 行业关联图特征
    # ==============================================
    def _industry_agg(self, x: np.ndarray, agg_func: str) -> np.ndarray:
        """
        按日期+行业分组聚合（向量化优化版）
        核心思路：利用行业ID(0-30)做mask，通过axis=1聚合+广播消除时间循环
        """
        T, N = x.shape
        result = np.full_like(x, np.nan)

        for ind_id in range(31):
            mask = (self.industry == ind_id)
            if not np.any(mask):
                continue  # 该行业无数据，跳过

            x_ind = np.where(mask, x, np.nan)

            if agg_func == 'median':
                agg_val = np.nanmedian(x_ind, axis=1, keepdims=True)
                result[mask] = np.broadcast_to(agg_val, (T, N))[mask]
            elif agg_func == 'mean':
                agg_val = np.nanmean(x_ind, axis=1, keepdims=True)
                result[mask] = np.broadcast_to(agg_val, (T, N))[mask]
            elif agg_func == 'std':
                agg_val = np.nanstd(x_ind, axis=1, keepdims=True, ddof=1)  # ddof=1对应样本标准差
                result[mask] = np.broadcast_to(agg_val, (T, N))[mask]
            elif agg_func == 'sum':
                agg_val = np.nansum(x_ind, axis=1, keepdims=True)
                result[mask] = np.broadcast_to(agg_val, (T, N))[mask]
            elif agg_func == 'rank':
                agg_rank = bn.nanrankdata(x_ind, axis=1)
                K = np.sum(mask, axis=1, keepdims=True)
                agg_rank = K + 1 - agg_rank
                agg_rank = agg_rank / K
                result[mask] = agg_rank[mask]
            else:
                raise ValueError(f"Unknown aggregation function: {agg_func}")

        return result

    def calc_peind(self) -> np.ndarray:
        """行业PE偏离：个股PE - 申万一级行业PE中位数"""
        ind_pe_median = self._industry_agg(self.pe, 'median')
        return self.pe - ind_pe_median

    def calc_qroe(self) -> np.ndarray:
        """ROE行业分位数：申万一级行业内ROE排名百分位数"""
        return self._industry_agg(self.roe, 'rank')

    def calc_deltapb(self) -> np.ndarray:
        """PB行业偏离度：(PB/行业PB中位数)-1"""
        ind_pb_median = self._industry_agg(self.pb, 'median')
        res = self.safe_div(self.pb, ind_pb_median)
        return res - 1

    def calc_cor_ret(self, window: int = 20) -> np.ndarray:
        """行业收益率相关性：个股与对应申万一级行业20日收益率相关系数（向量化优化）"""
        T, N = self.pct.shape
        ind_ret = np.full_like(self.pct, np.nan)

        for ind_id in range(31):
            mask = (self.industry == ind_id)
            if not np.any(mask):
                continue
            x_ind = np.where(mask, self.pct, np.nan)
            ind_mean = np.nanmean(x_ind, axis=1, keepdims=True)
            ind_ret[mask] = np.broadcast_to(ind_mean, (T, N))[mask]

        return self.rolling_nancorr(self.pct, ind_ret, window=window)

    def calc_dlev(self) -> np.ndarray:
        """资产负债率差异度：ABS(个股资产负债率-行业均值)/行业均值"""
        ind_debt_mean = self._industry_agg(self.debt_ratio, 'mean')
        res = self.safe_div(np.abs(self.debt_ratio - ind_debt_mean), ind_debt_mean)
        return res

    def calc_z(self) -> np.ndarray:
        """营收增速Z值：(个股营收增速-行业均值)/行业标准差"""
        ind_rev_mean = self._industry_agg(self.revenue_growth, 'mean')
        ind_rev_std = self._industry_agg(self.revenue_growth, 'std')
        z_score = self.safe_div(self.revenue_growth - ind_rev_mean, ind_rev_std)
        return z_score

    def calc_cfr(self) -> np.ndarray:
        """经营现金流行业占比：个股经营现金流/行业经营现金流总和"""
        ind_ocf_sum = self._industry_agg(self.ocf, 'sum')
        res = self.safe_div(self.ocf, ind_ocf_sum)
        return res


def calc_all_gat_features(gat: GAT_features) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    计算所有GAT模型特征
    返回：(节点特征字典, 边特征字典)
    """
    # 1. 共享节点特征
    shared_node = {
        'ret_20': gat.calc_ret_n(20),
        'ret_60': gat.calc_ret_n(60),
        'mom_120_20': gat.calc_mom_m_n(120, 20),
        'ret_std_20': gat.calc_ret_std_n(20),
        'ret_std_60': gat.calc_ret_std_n(60),
        'turn_std_10': gat.calc_turn_std_m(10),
        'turn_std_20': gat.calc_turn_std_m(20),
        'turn_10': gat.calc_turn_nm(10),
        'turn_20': gat.calc_turn_nm(20),
        'turn_60': gat.calc_turn_nm(60),
        'pospct_std_20': gat.calc_pospct_std(20),
        'pospct_std_60': gat.calc_pospct_std(60),
        'negpct_std_20': gat.calc_negpct_std(20),
        'negpct_std_60': gat.calc_negpct_std(60),
        'pct_vol_cor_20': gat.calc_pct_vol_cor(20),
        'pct_vol_cor_60': gat.calc_pct_vol_cor(60),
    }

    # 2. 资金流向图特征
    fund_node = {
        'aavg_20': gat.calc_aavg(20),
        'illiq_20': gat.calc_illiq(20),
        'stda_20': gat.calc_stda(20),
    }
    fund_edge = {
        'f': gat.calc_f(),
        'delta_tr_5': gat.calc_delta_tr(5),
        'cor_pv_20': gat.calc_cor_pv(20),
    }

    # 3. 行业关联图特征
    industry_node = {
        'peind': gat.calc_peind(),
        'qroe': gat.calc_qroe(),
        'deltapb': gat.calc_deltapb(),
    }
    industry_edge = {
        'cor_ret_20': gat.calc_cor_ret(20),
        'dlev': gat.calc_dlev(),
        'z': gat.calc_z(),
        'cfr': gat.calc_cfr(),
    }

    # 合并所有节点特征
    all_node_features = {**shared_node, **fund_node, **industry_node}
    # 合并所有边特征
    all_edge_features = {**fund_edge, **industry_edge}

    return all_node_features, all_edge_features



# def build_graph(edge_features: Dict[str, np.ndarray], t: int, corr_threshold: float = 0.6) -> np.ndarray:
#     """
#     构建第t日的图邻接矩阵
#     研报方法：对所有边特征的时序序列计算相关系数，等权求和后大于阈值则建边
#     """
#     N = edge_features[list(edge_features.keys())[0]].shape[1]
#     adj = np.eye(N, dtype=bool)  # 自环
#     # 提取所有边特征的历史80日序列
#     edge_seq = np.stack([feat[t - 79:t + 1] for feat in edge_features.values()], axis=-1)  # (80, N, 7)
#     # 计算每对股票的边特征相关系数
#     for i in range(N):
#         for j in range(i + 1, N):
#             # 计算7个边特征的相关系数并等权求和
#             corr_sum = 0.0
#             for k in range(7):
#                 corr = np.corrcoef(edge_seq[:, i, k], edge_seq[:, j, k])[0, 1]
#                 corr_sum += corr if not np.isnan(corr) else 0.0
#             avg_corr = corr_sum / 7
#
#             if avg_corr > corr_threshold:
#                 adj[i, j] = adj[j, i] = True
#     return adj


if __name__ == '__main__':

    basic_feats = load_basic_feats("/data/xujiayi/end2end/d_field")
    industry_mask = load_industry_mask('/data/xujiayi/end2end/mask/')
    fundamental_data = load_fundamental_feats('/data/xujiayi/end2end/PINN-MTICG/IndCapGAT/')

    gat = GAT_features(basic_feats, industry_mask, fundamental_data)
    node_feats, edge_feats = calc_all_gat_features(gat)
    all_feats = {**node_feats, **edge_feats}

    save_root = '/data/xujiayi/end2end/PINN-MTICG/IndCapGAT/'
    os.makedirs(save_root + "z_score", exist_ok=True)

    for k, v in all_feats.items():
        print(k)
        print(v)
        v.astype(float).tofile(f"{save_root}{k}.bin")

    fields = list(all_feats.keys())

    for f in fields:
        feat = np.memmap(f"{save_root}{f}.bin", shape=(3400, 5422), dtype=float, mode="r")
        mean = np.nanmean(feat, axis=1, keepdims=True)
        std = np.nanstd(feat, axis=1, keepdims=True)
        std = np.where((std < 1e-8) | np.isnan(std), 1.0, std)
        feat = (feat - mean) / std
        feat.astype(float).tofile(f"{save_root}/z_score/{f}_zscore.bin")



    fields = [
    'ret_20',
    'ret_60',
    'mom_120_20',
    'ret_std_20',
    'ret_std_60',
    'turn_std_10',
    'turn_std_20',
    'turn_10',
    'turn_20',
    'turn_60',
    'pospct_std_20',
    'pospct_std_60',
    'negpct_std_20',
    'negpct_std_60',
    'pct_vol_cor_20',
    'pct_vol_cor_60',

    'aavg_20',
    'illiq_20',
    'stda_20',

    'peind',
    'qroe',
    'deltapb',

    'f',
    'delta_tr_5',
    'cor_pv_20',
    'cor_ret_20',

    'dlev',
    'z',
    'cfr'
]
    for f in fields:
        feat = np.memmap(f"{save_root}/z_score/{f}_zscore.bin", shape=(3400, 5422), dtype=float, mode="r")
        print(feat)






