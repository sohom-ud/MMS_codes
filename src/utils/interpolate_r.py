'''
Clip and interpolate data so that the measurements of all 4 s/c have the same timestamps
'''
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import interp1d

def interpolate_r(data, res='B'):
    
    var_dict = {'B': 'b_gse', 'vi': 'v_spincorr_ion', 've': 'v_spincorr_elc'}
    var = var_dict[res]

    ms = data[f'{var}_1']['Epoch'][0]   # Start time of interval
    me = data[f'{var}_1']['Epoch'][-1]  # End time of interval

    dt = (data[f'{var}_1']['Epoch'][1] - data[f'{var}_1']['Epoch'][0]) # Resolution of magnetic field in days (Epochs are in Julian days)

    # Want to clip time series from all 4 s/c so that the start and end times match
    # Our approach here is to find the maximum start time and minimum end time for all 4 s/c.

    # Find minimum start time and maximum end time
    for probe in [1, 2, 3, 4]:
        
        max_start_time = max(ms, data[f'{var}_{probe}']['Epoch'][0])
        min_end_time = min(me, data[f'{var}_{probe}']['Epoch'][-1])

    # tnew = np.arange(max_start_time, min_end_time, dt)
    tnew = data[f'{var}_1']['Epoch'][np.logical_and(data[f'{var}_1']['Epoch']>max_start_time, data[f'{var}_1']['Epoch']<min_end_time)]

    for probe in [1, 2, 3, 4]:

        t = data[f'r_gse_{probe}']['Epoch']

        r = data[f'r_gse_{probe}']['Values']

        df = pd.DataFrame(data=np.column_stack([t, r]))
        df.set_index(0, inplace=True)

        df = df.drop_duplicates()
        
        # df.index = pd.to_datetime(df.index)
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])

        t = df.index
        r = df.values

        f = interp1d(t, r, axis=0, kind='quadratic', fill_value="extrapolate")

        # Resampling positions to tnew

        data[f'r_gse_{probe}_res_{res}'] = dict()

        data[f'r_gse_{probe}_res_{res}']['Epoch'] = tnew
        data[f'r_gse_{probe}_res_{res}']['Values']= f(tnew)