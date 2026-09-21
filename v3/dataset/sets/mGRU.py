import os
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v3.dataset.datapool import DataPool, ROOT
from v3.dataset.processor import *


output_dir = ROOT / "stock/model_input/mGRU"
output_dir.mkdir(parents=True, exist_ok=True)

data = DataPool(ROOT, asset="stock")
mclose = data.load('m_essentials/close').transpose(0,2,1)
mopen = data.load('m_essentials/open').transpose(0,2,1)
mhigh = data.load('m_essentials/high').transpose(0,2,1)
mlow = data.load('m_essentials/low').transpose(0,2,1)
mvolume = data.load('m_essentials/volume').transpose(0,2,1)

dopen = data.load('d_essentials/open')
ceil = data.load('basic/price_ceil')
floor = data.load('basic/price_floor')
scaler = data.load('d_essentials/adjusting_factor')


close2dopen = np.divide(
    mclose, dopen[:,:,np.newaxis],
    out=np.full_like(mclose,0.0),
    where=dopen[:,:,np.newaxis]!=0
)
close2dopen.transpose(0,2,1).astype(np.float32).tofile(output_dir / 'close2dopen.bin')

high2dopen = np.divide(
    mhigh, dopen[:,:,np.newaxis],
    out=np.full_like(mhigh,0.0),
    where=dopen[:,:,np.newaxis]!=0
)
high2dopen.transpose(0,2,1).astype(np.float32).tofile(output_dir / 'high2dopen.bin')

low2dopen = np.divide(
    mlow, dopen[:,:,np.newaxis],
    out=np.full_like(mlow,0.0),
    where=dopen[:,:,np.newaxis]!=0
)
low2dopen.transpose(0,2,1).astype(np.float32).tofile(output_dir / 'low2dopen.bin')

ppos = np.divide(
    mclose-floor[:,:,np.newaxis],
    ceil[:,:,np.newaxis]-floor[:,:,np.newaxis],
    out=np.full_like(mclose, 0.5),
    where=(ceil[:,:,np.newaxis]-floor[:,:,np.newaxis])!=0
)
ppos.transpose(0,2,1).astype(np.float32).tofile(output_dir / 'ppos.bin')
ppos


arr = np.divide(
    mvolume, scaler[:,:,np.newaxis],
    out=np.full_like(mvolume,np.nan), where=scaler[:,:,np.newaxis]!= 0,
)
arr_dailysum = bn.nansum(arr,axis=-1)   # T,N

arr_sw = np.lib.stride_tricks.sliding_window_view(arr_dailysum, 20, axis=0)
arr_rm = bn.nanmean(arr,axis=-1)  # T-w,N

arr2rollmean = np.divide(
    arr,  # T,N,241
    arr_rm[:,:,np.newaxis],
    out=np.full_like(arr, 0),
    where=arr_rm[:,:,np.newaxis]!=0
)
arr2rollmean.transpose(0,2,1).astype(np.float32).tofile(output_dir / 'volume_adj2rollmean.bin')
