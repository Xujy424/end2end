import numpy as np
import pandas as pd
import torch as th
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, List, Literal
import functools


# ==========================================
# 銆愮涓€灞傘€戝叕鍏卞熀绫伙細鎵€鏈夋暟鎹泦鍏辩敤锛堟棤宸紓浠ｇ爜锛?
# ==========================================
class BaseDataset(Dataset):
    root = Path('/data/xujiayi/xjy/')

    _axis_loaded = False
    _dates = None
    _ticks = None
    _tradable = None

    def __init__(self, start_date, end_date):
        # 鍔犺浇杞存暟鎹紙瀛愮被瀹炵幇锛?
        self.start_date = start_date
        self.end_date = end_date
        self._load_axis()

    def _post_init(self):
        # 鏃ユ湡杩囨护锛堝叕鍏遍€昏緫锛?
        self.valid_date_mask = (self.dates >= self.start_date) & (self.dates <= self.end_date)
        self.valid_date_indices = np.where(self.valid_date_mask)[0]

        # 鍔犺浇鐗瑰緛 & 鏍囩锛堝瓙绫诲疄鐜帮紝鍥犱负shape涓嶅悓锛?
        self._load_fields()
        self._load_labels()

        # 鍒濆鍖栨牱鏈槧灏?
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
    # 瀛愮被蹇呴』瀹炵幇鐨勬帴鍙ｏ紙宸紓閮ㄥ垎锛?
    # ------------------------------
    def _load_fields(self): raise NotImplementedError()
    def _load_labels(self): raise NotImplementedError()
    def _get_daily_feat(self, date_idx, tick_indices): raise NotImplementedError()
    def _get_label(self, date_idx, tick_indices): raise NotImplementedError()
    def _init_dataset(self): raise NotImplementedError()


# ==========================================
# 銆愮浜屽眰 A銆戞棩棰戝熀绫?
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
# 銆愮浜屽眰 B銆戝垎閽熼鍩虹被
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
        return feat #np.nan_to_num(feat, nan=0.0, copy=False) # N锛?37锛孎

    def _get_label(self, date_idx, tick_indices):
        return self.label_mmap[date_idx, tick_indices] if self.label else np.array([])

    def _init_dataset(self):
        pass


# ==========================================
# 銆愮涓夊眰銆?Single 涓氬姟鏁版嵁闆?Discard!
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
            freq: FreqType = "daily",  # 鏍稿績锛氶鐜囨爣璁?
            mode: str = "universe",
            fix_stock=None,
            pool_name = None,
            sample_size=None,
            lag=20,
    ):
        # 1. 鑷姩閫夋嫨搴曞眰鍩虹被锛堢粍鍚堟ā寮忥級
        self.freq = freq
        if freq == "daily":
            self.backend = DailyBase(data_path, fields, label, start_date, end_date, lag)  # N,L,F
        elif freq == "intraday":
            self.backend = IntradayBase(data_path, fields, label, start_date, end_date)    # N,237,F
        else:
            raise ValueError(f"涓嶆敮鎸侀鐜囷細{freq}锛屽彲閫?daily/intraday")
        self.dates = self.backend.dates
        self.ticks = self.backend.ticks
        self.valid_date_mask = self.backend.valid_date_mask

        # 2. 妯″紡閰嶇疆
        self.mode = mode
        self.fix_stock = fix_stock
        self.pool_name = pool_name
        self.sample_size = sample_size

        # 3. 缁熶竴鍒濆鍖栨牱鏈槧灏?
        self._init_dataset()

    def _init_dataset(self):
        self.data_map = {}

        # fix 妯″紡鑲＄エ绱㈠紩
        if self.mode == "fix":
            if not self.fix_stock:
                raise ValueError("fix 妯″紡蹇呴』浼犲叆 fix_stock")
            self.fix_tick_indices = [self.backend.ticks.index(t) for t in self.fix_stock]
        elif self.mode == 'pool':
            if not self.pool_name:
                raise ValueError("pool 妯″紡蹇呴』浼犲叆 pool_name")
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
            else: raise ValueError(f"涓嶆敮鎸佹ā寮忥細{self.mode}")
            
            for t in ticks:
                self.data_map[idx] = (d, t)
                idx += 1
        print(f"FlattenDataset 鏍锋湰鏁帮細{len(self.data_map)}")

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        d, t = self.data_map[idx]
        feat = self.backend._get_daily_feat(d, [t]).squeeze(0)  # [237, F] ,squeeze(0)
        label = self.backend._get_label(d, [t])[0] if self.backend.label else 0.0
        return th.from_numpy(feat).float(), th.tensor(label).float(), d, t


class BatchDataset(Dataset):
    """
    缁熶竴鎵规鏁版嵁闆嗭紙鏀寔鏃ラ / 鍒嗛挓棰戯級
    """
    def __init__(
        self,
        data_path,
        fields,
        label,
        start_date,
        end_date,
        freq: FreqType = "daily",  # 鏍稿績锛氶鐜囨爣璁?
        mode: str = "universe",
        fix_stock=None,
        pool_name = None,
        sample_size=None,
        lag=20,
    ):
        # 1. 鑷姩閫夋嫨搴曞眰鍩虹被锛堢粍鍚堟ā寮忥級 
        self.freq = freq
        if freq == "daily":
            self.backend = DailyBase(data_path, fields, label, start_date, end_date, lag)
        elif freq == "intraday":
            self.backend = IntradayBase(data_path, fields, label, start_date, end_date)
        else:
            raise ValueError(f"涓嶆敮鎸侀鐜囷細{freq}锛屽彲閫?daily/intraday")
        self.dates = self.backend.dates
        self.ticks = self.backend.ticks
        self.valid_date_mask = self.backend.valid_date_mask

        # 2. 鎵规妯″紡閰嶇疆锛堢粺涓€閫昏緫锛?
        self.mode = mode
        self.fix_stock = fix_stock
        self.pool_name = pool_name
        self.sample_size = sample_size

        # 3. 缁熶竴鍒濆鍖栨牱鏈槧灏?
        self._init_dataset()

    def _init_dataset(self):
        self.data_map: Dict[int, tuple] = {}
        self.date_str_map: Dict[int, str] = {}

        # fix 妯″紡鑲＄エ绱㈠紩
        if self.mode == "fix":
            if not self.fix_stock:
                raise ValueError("fix 妯″紡蹇呴』浼犲叆 fix_stock")
            self.fix_tick_indices = [self.backend.ticks.index(t) for t in self.fix_stock]
        elif self.mode == 'pool':
            if not self.pool_name:
                raise ValueError("pool 妯″紡蹇呴』浼犲叆 pool_name")
            self.pool = np.memmap(self.root/"mask"/f"{self.pool_name}_mask.bin", dtype=bool, mode="r", shape=(len(self.dates), len(self.ticks)))

        # 閬嶅巻鎵€鏈夋湁鏁堟棩鏈?
        current_idx = 0
        for date_idx in self.backend.valid_date_indices:
            date_str = self.backend.dates[date_idx]

            valid_mask = self.backend.tradable[date_idx] 
            if self.backend.label:
                valid_mask = valid_mask & ~np.isnan(self.backend.label_mmap[date_idx])

            # 1. 鏍规嵁妯″紡绛涢€夊綋鏃ヨ偂绁ㄧ殑鍏ㄥ眬绱㈠紩
            if self.mode == "universe":            # 褰撴棩鎵€鏈夊彲浜ゆ槗鑲＄エ锛堢洿鎺ュ彇鍏ㄥ眬绱㈠紩锛?
                tick_indices = np.where(valid_mask)[0]
            elif self.mode == 'pool':
                tick_indices = np.where(valid_mask & self.backend.pool[date_idx])[0]
            elif self.mode == "fix":
                tick_indices = np.array(self.fix_tick_indices)
            elif self.mode == "sample":        # 闅忔満鎶芥牱鍙氦鏄撹偂绁?
                if not self.sample_size:
                    raise ValueError("sample 妯″紡蹇呴』鎸囧畾 sample_size")
                valid_ticks = np.where(valid_mask)[0]
                if len(valid_ticks) < self.sample_size:
                    raise ValueError(f"{date_str} 鏈夋晥鑲＄エ涓嶈冻")
                tick_indices = np.array(sorted(random.sample(list(valid_ticks), self.sample_size)))
            else:
                raise ValueError(f"涓嶆敮鎸佹ā寮忥細{self.mode}")

            if len(tick_indices) == 0:
                continue

            # 2. 浠呰褰曠储寮曟槧灏勶紙涓嶅姞杞界壒寰侊紝鍒濆鍖栨瀬蹇級
            self.data_map[current_idx] = (date_idx, tick_indices)
            self.date_str_map[current_idx] = date_str
            current_idx += 1

        print(f"BatchDataset 灏辩华 | 棰戠巼={self.freq} | 鎵规={len(self.data_map)}")

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        d, t = self.data_map[idx]
        feat = self.backend._get_daily_feat(d, t)
        label = self.backend._get_label(d, t)

        valid_mask = (~np.isnan(feat).any(axis=(1, 2))) & (~np.isnan(label))   # 搴旇鏀规垚鏌愪釜鏃堕棿姝ヤ笂鐗瑰緛鍏ㄤ负闆?鎴?lable 涓洪浂鐨勮偂绁ㄤ笉涓簐alid
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


