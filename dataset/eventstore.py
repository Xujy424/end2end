# -*- coding:utf-8 -*-
"""
ERED 事件 Dataset。

和项目现有框架的关系：
- 继承 dataset.vanilla.BaseDataset；
- 提供 EventVecBase / EventMaskBase / EventAgeBase 三个 backend；
- 可以直接被 dataset.multicompose.MultiBatchDataset 通过 name=eventvec/eventmask/eventage 调用。

数据目录只需要三个文件：
    event_x.npy              float32 [E, D_event]
    event_tick.npy           int64   [E]
    event_effective_idx.npy  int64   [E]

输入事件表本身只包含真实财报事件，不补空季度。
因此 eventmask 不由数据处理阶段生成，而是在这里按“最近 K 个真实事件，不足 K 则 padding”动态生成。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .vanilla import BaseDataset


_EVENT_STORE_CACHE: Dict[Tuple[str, int], "PITEventStore"] = {}


def get_event_store(data_path, max_events):
    key = (str(Path(data_path)), int(max_events))
    if key not in _EVENT_STORE_CACHE:
        _EVENT_STORE_CACHE[key] = PITEventStore(data_path, max_events)
    return _EVENT_STORE_CACHE[key]


class PITEventStore:
    """Point-in-time 事件存储。

    对每个交易日 t 和股票 s，返回截至 t 已生效的最近 K 个真实事件。
    """

    def __init__(self, data_path, max_events=8):
        self.path = Path(data_path)
        self.max_events = int(max_events)

        self.event_x = np.load(self.path / "event_x.npy", mmap_mode="r")
        self.event_tick = np.load(self.path / "event_tick.npy", mmap_mode="r").astype(np.int64)
        self.event_effective_idx = np.load(self.path / "event_effective_idx.npy", mmap_mode="r").astype(np.int64)

        if self.event_x.ndim != 2:
            raise ValueError(f"event_x should be [E, D_event], got {self.event_x.shape}")
        if len(self.event_x) != len(self.event_tick) or len(self.event_x) != len(self.event_effective_idx):
            raise ValueError("event_x/event_tick/event_effective_idx length mismatch")

        self.event_dim = int(self.event_x.shape[1])
        axis_ticks = BaseDataset._ticks
        axis_num_ticks = len(axis_ticks) if axis_ticks is not None else 0
        event_num_ticks = int(self.event_tick.max()) + 1 if len(self.event_tick) else 0
        self.num_ticks = max(axis_num_ticks, event_num_ticks)

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
        """取横截面事件。

        Returns:
            eventvec:  [N, K, D_event]
            eventmask: [N, K]
            eventage:  [N, K]
        """
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
    """事件 backend 基类，与 vanilla.DailyBase 的接口保持一致。"""

    def __init__(self, data_path, start_date, end_date, max_events=8, include_today=True, **kwargs):
        super().__init__(start_date, end_date)
        self.data_path = Path(data_path)
        self.max_events = int(max_events)
        self.include_today = bool(include_today)
        self._post_init()

    def _load_fields(self):
        self.store = get_event_store(self.data_path, self.max_events)

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
    """返回 [N, K, D_event]。"""

    def _get_daily_feat(self, date_idx, tick_indices):
        x, _, _ = super()._get_daily_feat(date_idx, tick_indices)
        return x


class EventMaskBase(EventBase):
    """返回 [N, K]。"""

    def _get_daily_feat(self, date_idx, tick_indices):
        _, m, _ = super()._get_daily_feat(date_idx, tick_indices)
        return m


class EventAgeBase(EventBase):
    """返回 [N, K]。"""

    def _get_daily_feat(self, date_idx, tick_indices):
        _, _, age = super()._get_daily_feat(date_idx, tick_indices)
        return age


