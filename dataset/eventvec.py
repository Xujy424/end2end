from pathlib import Path
import pandas as pd
import numpy as np
import torch as th

from .vanilla import BaseDataset




# class EventVecBase(BaseDataset):

#     def __init__(self, data_path, event_ids, fields, start_date, end_date):
#         super().__init__(start_date, end_date)
#         self.data_path = Path(data_path)
#         self.event_ids = event_ids
#         self.fields = [f"event_{e}_{field}" for field in fields for e in self.event_ids]
#         self._post_init()

#     def _load_fields(self):
#         self.field_cache = {}
#         for f in self.fields:
#             path = self.data_path / f"{f}.bin"
#             self.field_cache[f] = np.memmap(path, dtype=float, mode='r', shape=(len(self.dates),len(self.ticks)))
    
#     def _get_daily_feat(self, date_idx, tick_indices):
#         feats = []
#         for f in self.fields:
#             val = self.field_cache[f][date_idx, tick_indices]
#             feats.append(val.T)   # N, F*K
#         feat = np.stack(feats, axis=-1).reshape(len(tick_indices),-1,len(self.event_ids)).transpose(0,2,1) # N,K,F
#         return np.nan_to_num(feat, nan=0.0, copy=False)

#     def _load_labels(self):
#         pass
#     def _init_dataset(self):
#         pass
#     def _get_label(self, date_idx, tick_indices):
#         pass



# class EventMaskBase(BaseDataset):
     
#     def __init__(self, data_path, event_ids, start_date, end_date):
#         super().__init__(start_date, end_date)
#         self.data_path = Path(data_path)
#         self.fields = [f"event_mask_{e}" for e in event_ids]
#         self._post_init()

#     def _load_fields(self):
#         self.field_cache = {}
#         for f in self.fields:
#             path = self.data_path / f"{f}.bin"
#             self.field_cache[f] = np.memmap(path, dtype=bool, mode='r', shape=(len(self.dates),len(self.ticks)))
    
#     def _get_daily_feat(self, date_idx, tick_indices):
#         feats = []
#         for f in self.fields:
#             val = self.field_cache[f][date_idx, tick_indices]
#             feats.append(val.T)   # N,
#         feat = np.stack(feats, axis=-1)   # N,K
#         return feat.astype(int)
    
#     def _load_labels(self):
#         pass
#     def _init_dataset(self):
#         pass
#     def _get_label(self, date_idx, tick_indices):
#         pass





class EventVecBase(BaseDataset):

    def __init__(self, data_path, event_ids, fields, start_date, end_date):
        super().__init__(start_date, end_date)
        self.data_path = Path(data_path)
        self.fields = fields
        self.event_ids = event_ids
        self._post_init()

    def _load_fields(self):
        self.feats = np.memmap(self.data_path/'event_vec.bin', dtype=float, mode='r', 
                               shape=(len(self.dates),len(self.ticks),len(self.event_ids),len(self.fields)))
    
    def _get_daily_feat(self, date_idx, tick_indices):
        feat = self.feats[date_idx, tick_indices]  # N,K,F
        return np.nan_to_num(feat, nan=0.0, copy=False)

    def _load_labels(self):
        pass
    def _init_dataset(self):
        pass
    def _get_label(self, date_idx, tick_indices):
        pass


class EventMaskBase(BaseDataset):
     
    def __init__(self, data_path, event_ids, start_date, end_date):
        super().__init__(start_date, end_date)
        self.data_path = Path(data_path)
        self.event_ids = event_ids
        self._post_init()

    def _load_fields(self):
        self.masks = np.memmap(self.data_path/'event_mask.bin', dtype=bool, mode='r', 
                               shape=(len(self.dates),len(self.ticks),len(self.event_ids)))
    
    def _get_daily_feat(self, date_idx, tick_indices):
        mask = self.masks[date_idx, tick_indices]  # N,K
        return mask.astype(int)
    
    def _load_labels(self):
        pass
    def _init_dataset(self):
        pass
    def _get_label(self, date_idx, tick_indices):
        pass
    