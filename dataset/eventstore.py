# -*- coding:utf-8 -*-
"""
ERED 事件数据集。

新版约定：事件数据目录只需要三个核心数组：
    event_x.npy               float32, [E, D_event]
    event_tick.npy            int64,   [E]
    event_effective_idx.npy   int64,   [E]

事件长表本身只保存真实发生/真实披露的财报事件，不保存“空季度”。
因此 padding、mask、最近 K 个事件的截取，都在 Dataset 按 date_idx 动态完成，
不要在数据处理阶段伪造空事件。

输出给模型：
    eventvec   [N, K, D_event]
    eventmask  [N, K]
    eventage   [N, K]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .vanilla import BaseDataset


_EVENT_STORE_CACHE: Dict[Tuple[str, int, int], "PITEventStore"] = {}


def get_event_store(data_path, max_events, num_ticks):
    key = (str(Path(data_path)), int(max_events), int(num_ticks))
    if key not in _EVENT_STORE_CACHE:
        _EVENT_STORE_CACHE[key] = PITEventStore(data_path, max_events=max_events, num_ticks=num_ticks)
    return _EVENT_STORE_CACHE[key]


class PITEventStore:
    """Point-in-time 财报事件存储。

    不依赖 metadata.json，也不依赖 event_tick_ptr.npy。
    num_ticks 由 vanilla.BaseDataset 的全市场股票轴传入。
    """

    def __init__(self, data_path, max_events=8, num_ticks=None):
        self.path = Path(data_path)
        self.max_events = int(max_events)

        self.event_x = np.load(self.path / "event_x.npy", mmap_mode="r")
        self.event_tick = np.load(self.path / "event_tick.npy", mmap_mode="r").astype(np.int64)
        self.event_effective_idx = np.load(self.path / "event_effective_idx.npy", mmap_mode="r").astype(np.int64)

        if self.event_x.ndim != 2:
            raise ValueError(f"event_x.npy 应为 [E, D_event]，实际 shape={self.event_x.shape}")
        if len(self.event_tick) != len(self.event_x) or len(self.event_effective_idx) != len(self.event_x):
            raise ValueError("event_x/event_tick/event_effective_idx 长度不一致")

        self.event_dim = int(self.event_x.shape[1])
        if num_ticks is None:
            self.num_ticks = int(self.event_tick.max()) + 1 if len(self.event_tick) else 0
        else:
            self.num_ticks = int(num_ticks)

        self._by_tick = self._build_index()
        self.cache_key = None
        self.cache_value = None

    def _build_index(self) -> List[np.ndarray]:
        by_tick: List[List[int]] = [[] for _ in range(self.num_ticks)]
        for event_idx, tick in enumerate(self.event_tick):
            tick = int(tick)
            if 0 <= tick < self.num_ticks:
                by_tick[tick].append(event_idx)

        out: List[np.ndarray] = []
        for indices in by_tick:
            if not indices:
                out.append(np.empty(0, dtype=np.int64))
                continue
            arr = np.asarray(indices, dtype=np.int64)
            order = np.argsort(self.event_effective_idx[arr], kind="mergesort")
            out.append(arr[order])
        return out

    def get_batch(self, trade_idx, tick_indices, include_today=True):
        """返回某交易日横截面的最近 K 个已知事件。"""

        tick_indices = np.asarray(tick_indices, dtype=np.int64)
        key = (int(trade_idx), tick_indices.tobytes(), bool(include_today))
        if key == self.cache_key:
            return self.cache_value

        n = len(tick_indices)
        k = self.max_events
        eventvec = np.zeros((n, k, self.event_dim), dtype=np.float32)
        eventmask = np.zeros((n, k), dtype=np.float32)
        eventage = np.full((n, k), -1, dtype=np.int64)
        side = "right" if include_today else "left"

        for i, tick in enumerate(tick_indices):
            if tick < 0 or tick >= self.num_ticks:
                continue
            idx_all = self._by_tick[int(tick)]
            if idx_all.size == 0:
                continue

            eff = self.event_effective_idx[idx_all]
            pos = np.searchsorted(eff, int(trade_idx), side=side)
            if pos <= 0:
                continue

            idx = idx_all[max(0, pos - k):pos]
            m = len(idx)
            eventvec[i, -m:] = np.nan_to_num(self.event_x[idx], nan=0.0, posinf=0.0, neginf=0.0)
            eventmask[i, -m:] = 1.0
            eventage[i, -m:] = int(trade_idx) - self.event_effective_idx[idx]

        self.cache_key = key
        self.cache_value = (eventvec, eventmask, eventage)
        return self.cache_value


class EventBase(BaseDataset):
    """事件特征 BaseDataset，供 MultiBatchDataset 组合调用。"""

    def __init__(self, data_path, start_date, end_date, max_events=8, include_today=True, **kwargs):
        super().__init__(start_date, end_date)
        self.data_path = Path(data_path)
        self.max_events = int(max_events)
        self.include_today = bool(include_today)
        self._post_init()

    def _load_fields(self):
        self.store = get_event_store(self.data_path, self.max_events, num_ticks=len(self.ticks))

    def _load_labels(self):
        pass

    def _init_dataset(self):
        pass

    def _get_daily_feat(self, date_idx, tick_indices):
        return self.store.get_batch(
            trade_idx=int(date_idx),
            tick_indices=tick_indices,
            include_today=self.include_today,
        )

    def _get_label(self, date_idx, tick_indices):
        return np.array([])


class EventVecBase(EventBase):
    def _get_daily_feat(self, date_idx, tick_indices):
        x, _, _ = super()._get_daily_feat(date_idx, tick_indices)
        return x


class EventMaskBase(EventBase):
    def _get_daily_feat(self, date_idx, tick_indices):
        _, m, _ = super()._get_daily_feat(date_idx, tick_indices)
        return m


class EventAgeBase(EventBase):
    def _get_daily_feat(self, date_idx, tick_indices):
        _, _, age = super()._get_daily_feat(date_idx, tick_indices)
        return age
