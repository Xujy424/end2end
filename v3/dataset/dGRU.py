import os
import sys
import numpy as np
from pathlib import Path

from datapool import DataPool, ROOT
from processor import *


d_fields = [
        'close_zscore','open_zscore','high_zscore','low_zscore','logvolume_zscore','turnover_zscore',
        'close_pct','open_pct','high_pct','low_pct','logvolume_pct','turnover_pct',
        'close2open','high2open','low2open','high2low','high2close','low2close',
    ]
m_fields = ['close2dopen','high2dopen','low2dopen','ppos','volume_adj2rollmean','amount2rollmean']

data = DataPool(ROOT, asset="stock")
dclose = data.load('d_essentials/close_adj')
dopen = data.load('d_essentials/open_adj')
dhigh = data.load('d_essentials/high_adj')
dlow = data.load('d_essentials/low_adj')
dvolume = data.load('d_essentials/volume_adj')
dturnover = data.load('d_essentials/turnover')


close_pct = winsorize(pct_change(dclose))
open_pct = winsorize(pct_change(dopen))
high_pct = winsorize(pct_change(dhigh))
low_pct = winsorize(pct_change(dlow))
logvolume_pct = winsorize(pct_change(np.log(dvolume)))
turnover_pct = winsorize(pct_change(dturnover))
