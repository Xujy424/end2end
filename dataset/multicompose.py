import numpy as np
import pandas as pd
import torch as th
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, List, Literal, Any

from .vanilla import DailyBase, IntradayBase
from .timecode import TimeCodeBase
from .eventstore import EventVecBase, EventMaskBase, EventAgeBase



shared_param_dict = {
        "start_date": "2013-01-01",
        "end_date": "2025-12-31",
        "label": "Yyeo.10D",
        "mode": "universe",
        "pool_name": None,
        "fix_stock": None,
        "sample_size": None,
}

specified_param_dict={
    'dailyset': {
        'data_path': '/data/xujiayi/xjy/d_field/',
        'fields': ['f1','f2','f3'], 
        'lag': 20,
    },
    'minuteset': {
        'data_path': '/data/xujiayi/xjy/m_field/',
        'fields': ['high2dopen','low2dopen','close2dopen','ppos','amount2ollmean','volume_adj2rollmean'],
    },
    'timecode':{
        'freq': 'intraday',
        'lag': None
    },
    'eventvec':{
        'data_path': '/data/xujiayi/xjy/research_factors/model_specific/ERED/',
        'event_ids': [4,239,387,257,476,237,546,520,144,77,55,223,293,487,415,512,151,365,355,195,146],
        'fields': ['roe','npm','gpm','l2a','nocf_yoy','basic_eps_yoy','net_profeit_yoy','operating_cost_yoy','np_parent_compay_owners_yoy','total_opearating_revenue_yoy','log_distance']
    },
    'eventmask':{
        'data_path': '/data/xujiayi/xjy/research_factors/model_specific/ERED/',
        'event_ids': [4,239,387,257,476,237,546,520,144,77,55,223,293,487,415,512,151,365,355,195,146],
    }
}


# ==========================================
# 第三层：Multi 聚合数据集
# ==========================================
class MultiFlattenDataset(Dataset):
    def __init__(self, shared_param_dict, specified_param_dict):
        self.backends = {}
        for name, cfg in specified_param_dict.items():
            if name == "dailyset":
                backend = DailyBase(
                    start_date=shared_param_dict['start_date'],
                    end_date=shared_param_dict['end_date'],
                    label=shared_param_dict['label'],
                    **cfg
                )
            elif name == "minuteset":
                backend = IntradayBase(
                    start_date=shared_param_dict['start_date'],
                    end_date=shared_param_dict['end_date'],
                    label=shared_param_dict['label'],
                    **cfg
                )
            elif name == 'timecode':
                backend = TimeCodeBase(
                    start_date=shared_param_dict['start_date'],
                    end_date=shared_param_dict['end_date'],
                    **cfg
                )
            else:
                raise ValueError(f"Unsupported dataset type: {name}")
            self.backends[name] = backend

        self.primary = list(self.backends.values())[0]
        self.dates = self.primary.dates
        self.ticks = self.primary.ticks
        self.valid_date_mask = self.primary.valid_date_mask
        # 股票索引映射，O(1) 查询，替代慢速 list.index
        self.tick2idx = {tick: idx for idx, tick in enumerate(self.ticks)}

        self.mode = shared_param_dict.get("mode", "universe")
        self.pool_name = shared_param_dict.get("pool_name")
        self.fix_stock = shared_param_dict.get("fix_stock")
        self.sample_size = shared_param_dict.get("sample_size")

        self._init_dataset()

    def _is_valid(self, all_feats: list[np.ndarray], base_valid: np.ndarray) -> np.ndarray:
        # 1. 提取形状信息
        shapes = [feat.shape for feat in all_feats]  # [(N,T1,F1), ...]
        Ts = [s[1] for s in shapes]
        Fs = [s[2] for s in shapes]
        max_T = max(Ts)
        total_F = sum(Fs)
        N = all_feats[0].shape[0]
        # 2. 预分配零填充数组 (N, max_T, sum_F)
        combined = np.zeros((N, max_T, total_F), dtype=all_feats[0].dtype)
        # 3. 拼接多分支特征
        col_start = 0
        for feat, T, F in zip(all_feats, Ts, Fs):
            combined[:, :T, col_start:col_start + F] = feat
            col_start += F
        # 4. NaN 有效性判断
        stock_invalid = np.all(np.isnan(combined), axis=-1).any(axis=1)
        return base_valid & ~stock_invalid

    def _init_dataset(self):
        # 用列表批量收集，最后一次性构造字典，规避频繁字典写入
        valid_date = []
        valid_tick = []
        print("[DEBUG] 进入 _init_dataset，开始初始化模式配置")

        if self.mode == "fix":
            if not self.fix_stock:
                raise ValueError("fix 模式必须传入 fix_stock")
            self.fix_tick_indices = np.array([self.tick2idx[t] for t in self.fix_stock])
        elif self.mode == 'pool':
            if not self.pool_name:
                raise ValueError("pool 模式必须传入 pool_name")
            self.pool_mask = np.memmap(
                self.primary.root / "mask" / f"{self.pool_name}_mask.bin",
                dtype=bool, mode="r",
                shape=(len(self.dates), len(self.ticks))
            )

        date_list = self.primary.valid_date_indices
        total_dates = len(date_list)
        print(f"[DEBUG] 开始遍历日期，总有效日期数: {total_dates}")

        for i, d in enumerate(date_list):
            if i % 100 == 0:
                print(f"[DEBUG] 已处理日期: {i}/{total_dates}")

            # 基础掩码，memmap 只读数组强制 copy
            base_valid = self.primary.tradable[d].copy()
            if self.primary.label:
                label_nan = ~np.isnan(self.primary.label_mmap[d].copy())
                base_valid &= label_nan
                if not np.any(base_valid):
                    continue

            # 筛选当日股票
            if self.mode == "universe":
                ticks = np.where(base_valid)[0]
            elif self.mode == "pool":
                pool_slice = self.pool_mask[d]
                ticks = np.where(base_valid & pool_slice)[0]
            elif self.mode == "fix":
                ticks = self.fix_tick_indices
            elif self.mode == "sample":
                ticks = np.where(base_valid)[0]
                ticks = np.array(sorted(random.sample(ticks.tolist(), self.sample_size)))
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")

            if ticks.size == 0:
                continue

            # 批量读取多分支特征
            # feat_list = [backend._get_daily_feat(d, ticks) for backend in self.backends.values()]
            # valid_mask = self._is_valid(feat_list, base_valid[ticks])
            d_feat = self.backends['dailyset']._get_daily_feat(d, ticks)
            valid_mask = ~np.all(np.isnan(d_feat), axis=-1).any(axis=1)
            valid_ticks = ticks[valid_mask]
            if valid_ticks.size == 0:
                continue

            # 向量化扩展，彻底消灭 Python 个股循环
            cnt = valid_ticks.shape[0]
            valid_date.extend(np.repeat(d, cnt).tolist())
            valid_tick.extend(valid_ticks.tolist())

        # 全局一次性生成 data_map
        self.data_map = {idx: (d, t) for idx, (d, t) in enumerate(zip(valid_date, valid_tick))}
        print(f"MultiFlattenDataset 样本数：{len(self.data_map)}")

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        d, t = self.data_map[idx]  # t 是单个股票索引 (int)
        label = self.primary._get_label(d, [t])
        label = th.tensor(label).float().contiguous()

        feats = {}
        for name, backend in self.backends.items():
            arr = backend._get_daily_feat(d, [t]).squeeze(0)  # (lag/237, F)
            arr = np.nan_to_num(arr, nan=0.0)
            feats[name] = th.from_numpy(arr).float().contiguous()

        return {
            'feats': feats,
            'label': label,
            'date_idx': d,
            'tick_idxs': t
        }


class MultiBatchDataset(Dataset):
    def __init__(self, shared_param_dict, specified_param_dict):
        self.backends = {}
        for name, cfg in specified_param_dict.items():
            if "dailyset" in name:
                backend = DailyBase(
                    start_date=shared_param_dict['start_date'],
                    end_date=shared_param_dict['end_date'],
                    label=shared_param_dict['label'],
                    **cfg
                )
            elif name == "minuteset":
                backend = IntradayBase(
                    start_date=shared_param_dict['start_date'],
                    end_date=shared_param_dict['end_date'],
                    label=shared_param_dict['label'],
                    **cfg
                )
            elif name == 'timecode':
                backend = TimeCodeBase(
                    start_date=shared_param_dict['start_date'],
                    end_date=shared_param_dict['end_date'],
                    **cfg
                )
            elif name == "eventvec":
                backend = EventVecBase(
                    start_date=shared_param_dict["start_date"],
                    end_date=shared_param_dict["end_date"],
                    **cfg
                )
            elif name == "eventmask":
                backend = EventMaskBase(
                    start_date=shared_param_dict["start_date"],
                    end_date=shared_param_dict["end_date"],
                    **cfg
                )
            elif name == "eventage":
                backend = EventAgeBase(
                    start_date=shared_param_dict["start_date"],
                    end_date=shared_param_dict["end_date"],
                    **cfg
                )
            else:
                raise ValueError(f"Unsupported dataset type: {name}")
            self.backends[name] = backend

        self.primary = list(self.backends.values())[0]
        self.dates = self.primary.dates
        self.ticks = self.primary.ticks
        self.valid_date_mask = self.primary.valid_date_mask
        # 股票索引映射，O(1) 查询
        self.tick2idx = {tick: idx for idx, tick in enumerate(self.ticks)}

        self.mode = shared_param_dict.get("mode", "universe")
        self.pool_name = shared_param_dict.get("pool_name")
        self.fix_stock = shared_param_dict.get("fix_stock")
        self.sample_size = shared_param_dict.get("sample_size")
        self.nanfilt_set = shared_param_dict.get('nanflit_set')
        self.backend_names = list(self.backends.keys())

        self._init_dataset()

    def _is_valid(self, all_feats: list[np.ndarray], base_valid: np.ndarray) -> np.ndarray:
        shapes = [feat.shape for feat in all_feats]  # [(N,T1,F1), ...]
        Ts = [s[1] for s in shapes]
        Fs = [s[2] for s in shapes]
        max_T = max(Ts)
        total_F = sum(Fs)
        N = all_feats[0].shape[0]
        combined = np.zeros((N, max_T, total_F), dtype=all_feats[0].dtype)
        col_start = 0
        for feat, T, F in zip(all_feats, Ts, Fs):
            combined[:, :T, col_start:col_start + F] = feat
            col_start += F
        stock_invalid = np.all(np.isnan(combined), axis=-1).any(axis=1)
        return base_valid & ~stock_invalid

    def _init_dataset(self):
        self.data_map = {}

        if self.mode == "fix":
            if not self.fix_stock:
                raise ValueError("fix 模式必须传入 fix_stock")
            self.fix_tick_indices = np.array([self.tick2idx[t] for t in self.fix_stock])
        elif self.mode == 'pool':
            if not self.pool_name:
                raise ValueError("pool 模式必须传入 pool_name")
            self.pool_mask = np.memmap(
                self.primary.root / "mask" / f"{self.pool_name}_mask.bin",
                dtype=bool, mode="r",
                shape=(len(self.dates), len(self.ticks))
            )

        idx = 0
        for i, d in enumerate(self.primary.valid_date_indices):
            # if i % 100 == 0:
            #     print(f"[DEBUG] 已处理日期: {i}/{len(self.primary.valid_date_indices)}")
            valid = self.primary.tradable[d].copy()
            if self.primary.label:
                label_nan = ~np.isnan(self.primary.label_mmap[d].copy())
                valid &= label_nan
                if not np.any(valid):
                    continue

            if self.mode == "universe":
                ticks = np.where(valid)[0]
            elif self.mode == "pool":
                pool_slice = self.pool_mask[d]
                ticks = np.where(valid & pool_slice)[0]
            elif self.mode == "fix":
                ticks = self.fix_tick_indices
            elif self.mode == "sample":
                ticks = np.where(valid)[0]
                ticks = np.array(sorted(random.sample(ticks.tolist(), self.sample_size)))
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")

            if ticks.size == 0:
                continue

            # feat_list = [backend._get_daily_feat(d, ticks) for backend in self.backends.values()]
            # valid_mask = self._is_valid(feat_list, valid[ticks])
            # valid_ticks = ticks[valid_mask]

            # if len(valid_ticks) > 0:
            #     self.data_map[idx] = (d, valid_ticks)
            #     idx += 1
            self.data_map[idx] = (d, ticks)
            idx += 1

        print(f"MultiBatchDataset 日期批次数：{len(self.data_map)}")

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        # d, t = self.data_map[idx]  # t: np.ndarray of tick indices

        # label = self.primary._get_label(d, t)
        # feats = {}
        # for name, backend in self.backends.items():
        #     arr = backend._get_daily_feat(d, t)
        #     arr = np.nan_to_num(arr, nan=0.0)
        #     feats[name] = th.from_numpy(arr).float().contiguous()

        # label_tensor = th.from_numpy(label).float().contiguous()
        # return {
        #     'feats': feats,
        #     'label': label_tensor,
        #     'date_idx': d,
        #     'tick_idxs': t
        # }
        d, candidate_ticks = self.data_map[idx]
        feat_list = [backend._get_daily_feat(d, candidate_ticks) for backend in self.backends.values()]
        label = self.primary._get_label(d, candidate_ticks)
        nanflit_feats = [feat for name, feat in zip(self.backend_names, feat_list) if name in self.nanfilt_set]
        valid_mask = self._is_valid(nanflit_feats,  np.ones(len(candidate_ticks), dtype=bool))
        
        final_ticks = candidate_ticks[valid_mask]
        label = label[valid_mask]
        feat_list = [feat[valid_mask] for feat in feat_list]

        feats = {}
        for name, feat_arr in zip(self.backends.keys(), feat_list):
            feat_arr = np.nan_to_num(feat_arr, nan=0.0)
            feats[name] = th.from_numpy(feat_arr).float().contiguous()

        label_tensor = th.from_numpy(label).float().contiguous()
        return {
            'feats': feats,
            'label': label_tensor,
            'date_idx': d,
            'tick_idxs': final_ticks
        }



def multi_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    feats_collated = {
        key: th.cat([
            sample['feats'][key].unsqueeze(0) if (sample['feats'][key].dim()==2) and (key in ['dailyset','minuteset']) else sample['feats'][key]
            for sample in batch
        ], dim=0)
        for key in batch[0]['feats']
    }

    labels = [sample['label'].reshape(-1) for sample in batch]
    label_collated = th.cat(labels)

    date_idx_collated = np.array([sample['date_idx'] for sample in batch])

    tick_idxs_samples = [np.asarray(sample['tick_idxs']).reshape(-1) for sample in batch]
    tick_idxs_collated = np.concatenate(tick_idxs_samples)

    return {
        'feats': feats_collated,
        'label': label_collated,
        'date_idx': date_idx_collated,
        'tick_idxs': tick_idxs_collated,
    }




# ==========================================
# DataLoader 运行样例（使用模拟数据）
# ==========================================
if __name__ == "__main__":
        
        import os
        from tqdm import tqdm

        # 配置字典
        shared_param_dict = {
            "start_date": "2013-01-01",
            "end_date": "2025-12-31",
            "label": "Yyeo.10D",
            "mode": "pool",
            "pool_name": 'zzfull',
            "fix_stock": None,
            "sample_size": None,
        }

        specified_param_dict={
            'dailyset': {
                'data_path': '/data/xujiayi/end2end/GRU_new/',
                'fields': [
                    f.split('.')[0] for f in os.listdir('/data/xujiayi/end2end/GRU_new/') 
                    if not f.startswith('Y') and not any(sub in f for sub in ['tradable','date','tick','zzfull_mask'])
                ], 
                'lag': 20,
            },
            'minuteset': {
                'data_path': '/data/xujiayi/end2end/m_field/',
                'fields': ['close2dopen','high2dopen','low2dopen','ppos','volume_adj2rollmean','amount2rollmean'],
            },
            'timecode':{
                'freq': 'intraday',
                'lag': None
            }
        }

        # Flatten 模式测试
        print("\n=== MultiFlattenDataset ===")
        multi_flat = MultiFlattenDataset(shared_param_dict, specified_param_dict)
        loader_flat = DataLoader(multi_flat, batch_size=4, shuffle=False, num_workers=0, collate_fn=multi_collate_fn)
        for batch_dict in tqdm(loader_flat):
            feats, label, d, t = batch_dict.values()
            print(f"特征路数: {len(feats)}")
            for i, f in feats.items():
                print(f"  特征 {i} shape: {f.shape}")
            print(f"  标签 shape: {label.shape}")

        # # Batch 模式测试
        # print("\n=== MultiBatchDataset ===")
        # multi_batch = MultiBatchDataset(shared_param_dict, specified_param_dict)
        # loader_batch = DataLoader(multi_batch, batch_size=1, shuffle=False, num_workers=0, collate_fn=multi_collate_fn)
        # for batch_dict in loader_batch:
        #     feats, label, d, t = batch_dict.values()
        #     print(f"特征路数: {len(feats)}")
        #     for i, f in feats.items():
        #         print(f"  特征{i} shape: {f.shape}")
        #     print(f"  标签 shape: {label.shape}")
        #     break