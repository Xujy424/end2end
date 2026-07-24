import numpy as np
import pandas as pd
import torch as th
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, List, Literal
import functools


# ==========================================
# 【第一层】公共基类：所有数据集共用（无差异代码）
# ==========================================
class BaseDataset(Dataset):
    root = Path('/data/xujiayi/xjy/')

    _axis_loaded = False
    _dates = None
    _ticks = None
    _tradable = None

    def __init__(self, start_date, end_date):
        # 加载轴数据（子类实现）
        self.start_date = start_date
        self.end_date = end_date
        self._load_axis()

    def _post_init(self):
        # 日期过滤（公共逻辑）
        self.valid_date_mask = (self.dates >= self.start_date) & (self.dates <= self.end_date)
        self.valid_date_indices = np.where(self.valid_date_mask)[0]

        # 加载特征 & 标签（子类实现，因为shape不同）
        self._load_fields()
        self._load_labels()

        # 初始化样本映射
        self._init_dataset()

    @functools.lru_cache(maxsize=1)
    def _load_axis(self): 
        if not BaseDataset._axis_loaded:
            BaseDataset._dates = pd.to_datetime(np.load(self.root/"axis"/"dates.npy", allow_pickle=True)).strftime("%Y-%m-%d")
            BaseDataset._ticks = np.load(self.root/"axis"/"ticks.npy", allow_pickle=True)
            BaseDataset._tradable = np.memmap(self.root/"mask"/"tradable.bin", dtype=bool, mode="r", shape=(len(BaseDataset._dates), len(BaseDataset._ticks)))
            BaseDataset._axis_loaded = True
        self.dates = BaseDataset._dates
        self.ticks = BaseDataset._ticks
        self.tradable = BaseDataset._tradable
    # ------------------------------
    # 子类必须实现的接口（差异部分）
    # ------------------------------
    def _load_fields(self): raise NotImplementedError()
    def _load_labels(self): raise NotImplementedError()
    def _get_daily_feat(self, date_idx, tick_indices): raise NotImplementedError()
    def _get_label(self, date_idx, tick_indices): raise NotImplementedError()
    def _init_dataset(self): raise NotImplementedError()


# ==========================================
# 【第二层 A】日频基类
# ==========================================
class DailyBase(BaseDataset):

    def __init__(self, data_path, fields, label, start_date, end_date, lag):
        super().__init__(start_date, end_date)
        self.lag = lag
        self.data_path = Path(data_path)
        self.fields = fields
        self.label = label
        self._post_init()

    def _load_fields(self):
        self.field_cache = {}
        for f in self.fields:
            path = self.data_path / f"{f}.bin"
            self.field_cache[f] = np.memmap(path, dtype=float, mode="r", shape=(len(self.dates), len(self.ticks)))

    def _load_labels(self):
        if self.label:
            self.label_mmap = np.memmap(self.root/"label"/f"{self.label}.bin", dtype=float, mode="r", shape=(len(self.dates), len(self.ticks)))

    def _get_daily_feat(self, date_idx, tick_indices):
        feats = []
        for f in self.fields:
            val = self.field_cache[f][date_idx-self.lag+1:date_idx+1, tick_indices]
            feats.append(val.T)
        feat = np.stack(feats, axis=-1)
        return feat # np.nan_to_num(feat, nan=0.0, copy=False)

    def _get_label(self, date_idx, tick_indices):
        return self.label_mmap[date_idx, tick_indices] if self.label else np.array([])

    def _init_dataset(self):
        pass


# ==========================================
# 【第二层 B】分钟频基类
# ==========================================
class IntradayBase(BaseDataset):
    def __init__(self, data_path, fields, label, start_date, end_date):
        super().__init__(start_date, end_date)
        self.data_path = Path(data_path)
        self.fields = fields
        self.label = label
        self._post_init()

    def _load_fields(self):
        self.field_cache = {}
        for f in self.fields:
            path = self.data_path / f"{f}.bin"
            self.field_cache[f] = np.memmap(path, dtype=float, mode="r", shape=(len(self.dates), len(self.ticks), 237))

    def _load_labels(self):
        if self.label:
            self.label_mmap = np.memmap(self.root/f"label/{self.label}.bin", dtype=float, mode="r", shape=(len(self.dates), len(self.ticks)))

    def _get_daily_feat(self, date_idx, tick_indices):
        feats = []
        for f in self.fields:
            val = self.field_cache[f][date_idx][tick_indices]
            feats.append(val)
        feat = np.stack(feats, axis=-1)
        return feat #np.nan_to_num(feat, nan=0.0, copy=False) # N，237，F

    def _get_label(self, date_idx, tick_indices):
        return self.label_mmap[date_idx, tick_indices] if self.label else np.array([])

    def _init_dataset(self):
        pass


# ==========================================
# 【第三层】 Single 业务数据集 Discard!
# ==========================================
FreqType = Literal["daily", "intraday"]


class FlattenDataset(Dataset):
    def __init__(
            self,
            data_path,
            fields,
            label,
            start_date,
            end_date,
            freq: FreqType = "daily",  # 核心：频率标记
            mode: str = "universe",
            fix_stock=None,
            pool_name = None,
            sample_size=None,
            lag=20,
    ):
        # 1. 自动选择底层基类（组合模式）
        self.freq = freq
        if freq == "daily":
            self.backend = DailyBase(data_path, fields, label, start_date, end_date, lag)  # N,L,F
        elif freq == "intraday":
            self.backend = IntradayBase(data_path, fields, label, start_date, end_date)    # N,237,F
        else:
            raise ValueError(f"不支持频率：{freq}，可选 daily/intraday")
        self.dates = self.backend.dates
        self.ticks = self.backend.ticks
        self.valid_date_mask = self.backend.valid_date_mask

        # 2. 模式配置
        self.mode = mode
        self.fix_stock = fix_stock
        self.pool_name = pool_name
        self.sample_size = sample_size

        # 3. 统一初始化样本映射
        self._init_dataset()

    def _init_dataset(self):
        self.data_map = {}

        # fix 模式股票索引
        if self.mode == "fix":
            if not self.fix_stock:
                raise ValueError("fix 模式必须传入 fix_stock")
            self.fix_tick_indices = [self.backend.ticks.index(t) for t in self.fix_stock]
        elif self.mode == 'pool':
            if not self.pool_name:
                raise ValueError("pool 模式必须传入 pool_name")
            self.pool = np.memmap(self.root/"mask"/f"{self.pool_name}_mask.bin", dtype=bool, mode="r", shape=(len(self.dates), len(self.ticks)))

        idx = 0
        for d in self.backend.valid_date_indices:

            valid = self.backend.tradable[d]
            if self.backend.label:
                valid = valid & ~np.isnan(self.backend.label_mmap[d])

            if self.mode=='universe': ticks = np.where(valid)[0]
            elif self.mode=='pool': ticks = np.where(valid & self.backend.pool[d])[0]
            elif self.mode=="fix": ticks = np.array(self.fix_tick_indices)
            elif self.mode == "sample":
                ticks = np.where(valid)[0]
                ticks = np.array(sorted(random.sample(list(ticks), self.sample_size)))
            else: raise ValueError(f"不支持模式：{self.mode}")
            
            for t in ticks:
                self.data_map[idx] = (d, t)
                idx += 1
        print(f"FlattenDataset 样本数：{len(self.data_map)}")

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        d, t = self.data_map[idx]
        feat = self.backend._get_daily_feat(d, [t]).squeeze(0)  # [237, F] ,squeeze(0)
        label = self.backend._get_label(d, [t])[0] if self.backend.label else 0.0
        return th.from_numpy(feat).float(), th.tensor(label).float(), d, t


class BatchDataset(Dataset):
    """
    统一批次数据集（支持日频 / 分钟频）
    """
    def __init__(
        self,
        data_path,
        fields,
        label,
        start_date,
        end_date,
        freq: FreqType = "daily",  # 核心：频率标记
        mode: str = "universe",
        fix_stock=None,
        pool_name = None,
        sample_size=None,
        lag=20,
    ):
        # 1. 自动选择底层基类（组合模式） 
        self.freq = freq
        if freq == "daily":
            self.backend = DailyBase(data_path, fields, label, start_date, end_date, lag)
        elif freq == "intraday":
            self.backend = IntradayBase(data_path, fields, label, start_date, end_date)
        else:
            raise ValueError(f"不支持频率：{freq}，可选 daily/intraday")
        self.dates = self.backend.dates
        self.ticks = self.backend.ticks
        self.valid_date_mask = self.backend.valid_date_mask

        # 2. 批次模式配置（统一逻辑）
        self.mode = mode
        self.fix_stock = fix_stock
        self.pool_name = pool_name
        self.sample_size = sample_size

        # 3. 统一初始化样本映射
        self._init_dataset()

    def _init_dataset(self):
        self.data_map: Dict[int, tuple] = {}
        self.date_str_map: Dict[int, str] = {}

        # fix 模式股票索引
        if self.mode == "fix":
            if not self.fix_stock:
                raise ValueError("fix 模式必须传入 fix_stock")
            self.fix_tick_indices = [self.backend.ticks.index(t) for t in self.fix_stock]
        elif self.mode == 'pool':
            if not self.pool_name:
                raise ValueError("pool 模式必须传入 pool_name")
            self.pool = np.memmap(self.root/"mask"/f"{self.pool_name}_mask.bin", dtype=bool, mode="r", shape=(len(self.dates), len(self.ticks)))

        # 遍历所有有效日期
        current_idx = 0
        for date_idx in self.backend.valid_date_indices:
            date_str = self.backend.dates[date_idx]

            valid_mask = self.backend.tradable[date_idx] 
            if self.backend.label:
                valid_mask = valid_mask & ~np.isnan(self.backend.label_mmap[date_idx])

            # 1. 根据模式筛选当日股票的全局索引
            if self.mode == "universe":            # 当日所有可交易股票（直接取全局索引）
                tick_indices = np.where(valid_mask)[0]
            elif self.mode == 'pool':
                tick_indices = np.where(valid_mask & self.backend.pool[date_idx])[0]
            elif self.mode == "fix":
                tick_indices = np.array(self.fix_tick_indices)
            elif self.mode == "sample":        # 随机抽样可交易股票
                if not self.sample_size:
                    raise ValueError("sample 模式必须指定 sample_size")
                valid_ticks = np.where(valid_mask)[0]
                if len(valid_ticks) < self.sample_size:
                    raise ValueError(f"{date_str} 有效股票不足")
                tick_indices = np.array(sorted(random.sample(list(valid_ticks), self.sample_size)))
            else:
                raise ValueError(f"不支持模式：{self.mode}")

            if len(tick_indices) == 0:
                continue

            # 2. 仅记录索引映射（不加载特征，初始化极快）
            self.data_map[current_idx] = (date_idx, tick_indices)
            self.date_str_map[current_idx] = date_str
            current_idx += 1

        print(f"BatchDataset 就绪 | 频率={self.freq} | 批次={len(self.data_map)}")

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        d, t = self.data_map[idx]
        feat = self.backend._get_daily_feat(d, t)
        label = self.backend._get_label(d, t)

        valid_mask = (~np.isnan(feat).any(axis=(1, 2))) & (~np.isnan(label))   # 应该改成某个时间步上特征全为零 或 lable 为零的股票不为valid
        label = label[valid_mask]
        t = t[valid_mask]
        feat = feat[valid_mask]
        feat = np.nan_to_num(feat, nan=0.0, copy=False)

        feat_tensor = th.from_numpy(feat).float().contiguous()
        label_tensor = th.from_numpy(label).float().contiguous() if label.size > 0 else th.empty(0, dtype=th.float32)
        t = th.from_numpy(t).long()
        return feat_tensor, label_tensor, d, t


# def collate_fn(batch):
#     xs = th.stack([x[0] for x in batch])
#     ys = th.stack([x[1] for x in batch]) if batch[0][1].numel() > 0 else th.tensor([])
#     ds = np.array([x[2] for x in batch])
#     ts = np.array([x[3] for x in batch])
#     return xs, ys, ds, ts


