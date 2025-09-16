'''
Clips data to specified time range
'''

import numpy as np
import pandas as pd

def time_clip(data, trange):

    start_time = pd.Timestamp(trange[0]).to_julian_date()
    end_time = pd.Timestamp(trange[1]).to_julian_date()
   
    for key, value in data.items():
        
        print(key)
        # epoch = data[key].index
        epoch = data[key]['Epoch']

        cond = np.logical_and(epoch>=start_time, epoch<=end_time)

        data[key]['Epoch'] = data[key]['Epoch'][cond]
        data[key]['Values'] = data[key]['Values'][cond]