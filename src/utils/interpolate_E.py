'''
Interpolate the magnetic field measurements from all 4 s/c to the same time range
'''

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import interp1d

def interpolate_E(data):

    ms = data[f'b_gse_1']['Epoch'][0]
    me = data[f'b_gse_1']['Epoch'][-1]

    dt = (data['b_gse_1']['Epoch'][1] - data['b_gse_1']['Epoch'][0])# Time resolution of magnetic field

    # Find minimum start time and maximum end time
    for probe in [1, 2, 3, 4]:
        
        max_start_time = max(ms, data[f'b_gse_{probe}']['Epoch'][0])
        min_end_time = min(me, data[f'b_gse_{probe}']['Epoch'][-1])

    for probe in [1, 2, 3, 4]:

        t = data[f'edp_dce_gse_{probe}']['Epoch'][...]
        E = data[f'edp_dce_gse_{probe}']['Values'][...]
        
        df = pd.DataFrame(data=np.column_stack([t, E]))
        df.set_index(0, inplace=True)

        for col in df.columns:
            df[col] = pd.to_numeric(df[col])

        df = df.drop_duplicates()

        # df.index = pd.to_datetime(df.index)

        t = df.index
        E = df.values

        f = interp1d(t, E, axis=0, kind='quadratic', fill_value="extrapolate")
        
        # tnew = np.arange(max_start_time, min_end_time, dt)
        tnew = data['edp_dce_gse_1']['Epoch'][np.logical_and(data['edp_dce_gse_1']['Epoch']>max_start_time, data['edp_dce_gse_1']['Epoch']<min_end_time)]
        
        data[f'edp_dce_gse_{probe}']['Epoch'] = tnew
        data[f'edp_dce_gse_{probe}']['Values'] = f(tnew)
