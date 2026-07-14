import numpy as np
import os
from typing import Dict

from ..utils import Processor, Calculator, Loader



class GRU_features(Processor, Calculator):
    '''
        features for Transformer in PINN-MTICG
    '''
    def __init__(self, basic_features: Dict[str, np.ndarray], basic_masks: Dict[str, np.ndarray]):
        self.open = basic_features['open_adj']
        self.high = basic_features['high_adj']
        self.low = basic_features['low_adj']
        self.close = basic_features['close_adj']
        self.volume = basic_features['volume_adj']
        self.amount = basic_features['amount']
        self.turnover = basic_features['turnover']

        self.dict = basic_features

        self.industry_mask = basic_masks['industry']
        self.logmv = basic_features['logmv']

    def calc_ratio(self):
        self.dict['close2open'] = self.safe_div(self.close, self.open)
        self.dict['close2high'] = self.safe_div(self.close, self.high)
        self.dict['close2low'] = self.safe_div(self.close, self.low)
        self.dict['high2low'] = self.safe_div(self.high, self.low)
        self.dict['high2open'] = self.safe_div(self.high, self.open)
        self.dict['low2open'] = self.safe_div(self.low, self.open)
        print("Ratio features calculated.")
    
    def calc_diff(self, lags=[1,5,20,60]):
        fields = list(self.dict.keys())
        fields.remove('logmv')
        for l in lags:
            for f in fields:
                arr_lag = np.roll(self.dict[f], shift=l, axis=0)
                arr_lag[:l] = np.nan
                self.dict[f'{f}_diff{l}'] = self.safe_div(self.dict[f]-arr_lag, arr_lag)
        print("Diff features calculated.")
    
    def summarize(self):
        for k,v in self.dict.items():
            self.dict[k] = self.yeojohnson(v)
        print("Features yeojohnson completed.")

    def calc_label(self, lags=[1,5,10,20]):
        for l in lags:
            y = np.full_like(self.close, np.nan)
            y[:-(l+1)] = np.divide(self.close[l+1:], self.close[1:-l], out=np.full_like(self.close[l+1:], 0), where=self.close[1:-l]!=0)
            self.dict[f'Y.{l}D'] = y
            y_yeo = self.yeojohnson(y)
            self.dict[f'Yyeo.{l}D'] = y_yeo
            y_z = self.cross_standardize(y)
            self.dict[f'Yz.{l}D'] = y_z
            y_dm = self.indmv_neutral_longshort(y, self.industry_mask, self.logmv)
            self.dict[f'Ydm.{l}D'] = y_dm
            y_r = self.rank_transform(y)
            self.dict[f'Yr.{l}D'] = y_r
            y_ls = self.winsorize_linearsmooth(y)
            self.dict[f'Yls.{l}D'] = y_ls
        print("Labels calculated.")

   

if __name__ == '__main__':

    loader = Loader('/data/xujiayi/xjy/')

    os.makedirs('/data/xujiayi/end2end/dGRU', exist_ok=True)

    basic_feats = loader.load_daily_feats()
    basic_masks = loader.load_mask()

    data = GRU_features(basic_feats, basic_masks)
    data.calc_ratio()
    data.calc_diff()
    data.summarize()
    # data.calc_label()

    for k,v in data.dict.items():
        v.astype('float').tofile(f'/data/xujiayi/end2end/dGRU/{k}.bin')



    # dates = np.load('/data/xujiayi/end2end/axis/dates.npy',allow_pickle=True)
    # ticks = np.load('/data/xujiayi/end2end/axis/ticks.npy',allow_pickle=True)

    # os.makedirs('/data/xujiayi/end2end/mGRU', exist_ok=True)

    # m_fields = ['close_adj','high_adj','low_adj','open_adj','volume_adj','amount']
    # for feat in m_fields:
    #     arr = np.memmap(f'/data/xujiayi/end2end/m_field/{feat}.bin', dtype='float', mode='r', shape=(len(dates),len(ticks),241))
        
    #     arr1 = arr.transpose(2,0,1)[1:-3,:,:].transpose(1,2,0)
    #     mean = np.nanmean(arr1,axis=-1,keepdims=True)
    #     std = np.nanstd(arr1,axis=-1,keepdims=True)
    #     arr_z = np.divide(arr1 - mean, std, out=np.full_like(arr1, 0), where=(std)!=0)
    #     arr_z.astype('float').tofile(f'/data/xujiayi/end2end/mGRU/{feat}_Tz.bin')
        
    #     arr2 = arr.transpose(2,0,1)
    #     arr2 = np.divide(arr2[1:,:,:], arr2[:-1,:,:], out=np.full_like(arr2[1:,:,:], 0), where=arr2[:-1,:,:]!=0) -1
    #     arr2 = arr2[:-3,:,:].transpose(1,2,0)
    #     arr2.astype('float').tofile(f'/data/xujiayi/end2end/mGRU/{feat}_diff.bin')

    # for feat in ['volume_adj2rollmean','ppos','low2dopen','high2dopen','close2dopen','amount2rollmean','volume_adj']:
    #     arr = np.memmap(f'/data/xujiayi/end2end/m_field/{feat}.bin', dtype='float', mode='r', shape=(len(dates),len(ticks),241))
    #     if feat!='volume_adj':
    #         arr = arr.transpose(2,0,1)[1:-3,:,:].transpose(1,2,0)
    #         arr.astype('float').tofile(f'/data/xujiayi/end2end/mGRU/{feat}.bin')
    #     else:
    #         arr = np.log(arr)

    #         arr1 = arr.transpose(2,0,1)
    #         arr1 = np.divide(arr1[1:,:,:], arr1[:-1,:,:], out=np.full_like(arr1[1:,:,:], 0), where=arr1[:-1,:,:]!=0) -1
    #         arr1 = arr1[:-3,:,:].transpose(1,2,0)
    #         arr1.astype('float').tofile(f'/data/xujiayi/end2end/mGRU/log_{feat}_diff.bin')

    #         arr2 = arr.transpose(2,0,1)[1:-3,:,:].transpose(1,2,0)
    #         mean = np.nanmean(arr2,axis=-1,keepdims=True)
    #         std = np.nanstd(arr2,axis=-1,keepdims=True)
    #         arr_z = np.divide(arr2 - mean, std, out=np.full_like(arr2, 0), where=(std)!=0)
    #         arr_z.astype('float').tofile(f'/data/xujiayi/end2end/mGRU/log_{feat}_Tz.bin')