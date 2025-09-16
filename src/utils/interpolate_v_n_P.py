'''
Interpolate the velocities, densities and pressure tensor so that the timestamps match for all 4 spacecraft
'''

import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from scipy.interpolate import interp1d

def interpolate_v_n_P(data, species='ion'):
   
    ms = data[f'v_{species}_1']['Epoch'][0]
    me = data[f'v_{species}_1']['Epoch'][-1]

    dt = (data['v_ion_1']['Epoch'][1] - data['v_ion_1']['Epoch'][0]) if species =='ion' else (data['v_elc_1']['Epoch'][1] - data['v_elc_1']['Epoch'][0]) # Time resolution of moments in seconds

    start_time = pd.to_datetime(ms, origin='julian', unit='D')
    end_time = pd.to_datetime(ms, origin='julian', unit='D')

    bad_start = datetime(2018, 6, 1)

    if species=='elc' and (start_time>bad_start or end_time>bad_start):
        probe_list = [1, 2, 3]
    else:
        probe_list = [1, 2, 3, 4]

    # Find minimum start time and maximum end time
    for probe in probe_list:
        
        max_start_time = max(ms, data[f'v_{species}_{probe}']['Epoch'][0])
        min_end_time = min(me, data[f'v_{species}_{probe}']['Epoch'][-1])

    for var in ['v', 'v_spin', 'v_spincorr', 'N', 'Ptensor', 'Temptensor']:

        for probe in probe_list:

            t = data[f'{var}_{species}_{probe}']['Epoch']
            v = data[f'{var}_{species}_{probe}']['Values']
            
            if var == 'Ptensor' or var == 'Temptensor':

                v = data[f'{var}_{species}_{probe}']['Values'].reshape(len(t), 9)

            df = pd.DataFrame(data=np.column_stack([t, v]))
            df.set_index(0, inplace=True)

            df = df.drop_duplicates()

            t = df.index
            v = df.values

            f = interp1d(t, v, axis=0, kind='quadratic', fill_value="extrapolate")

            # tnew = np.arange(max_start_time, min_end_time, dt)
            tnew = data[f'{var}_{species}_1']['Epoch'][np.logical_and(data[f'{var}_{species}_1']['Epoch']>max_start_time, data[f'{var}_{species}_1']['Epoch']<min_end_time)]

            data[f'{var}_{species}_{probe}']['Epoch'] = tnew
            data[f'{var}_{species}_{probe}']['Values'] = f(tnew)


def interpolate_v_n_P_after_201806(data, species='ion'):
   
    ms = data[f'v_{species}_1']['Epoch'][0]
    me = data[f'v_{species}_1']['Epoch'][-1]

    dt = (data['v_ion_1']['Epoch'][1] - data['v_ion_1']['Epoch'][0]) if species =='ion' else (data['v_elc_1']['Epoch'][1] - data['v_elc_1']['Epoch'][0]) # Time resolution of moments in seconds

    # Find minimum start time and maximum end time
    for probe in [1, 2, 3]:
        
        max_start_time = max(ms, data[f'v_{species}_{probe}']['Epoch'][0])
        min_end_time = min(me, data[f'v_{species}_{probe}']['Epoch'][-1])

    for var in ['v', 'v_spin', 'v_spincorr', 'N', 'Ptensor']:

        for probe in [1, 2, 3]:

            t = data[f'{var}_{species}_{probe}']['Epoch']
            v = data[f'{var}_{species}_{probe}']['Values']
            
            if var == 'Ptensor':

                v = data[f'{var}_{species}_{probe}']['Values'].reshape(len(t), 9)

            df = pd.DataFrame(data=np.column_stack([t, v]))
            df.set_index(0, inplace=True)

            df = df.drop_duplicates()

            t = df.index
            v = df.values

            f = interp1d(t, v, axis=0, kind='quadratic', fill_value="extrapolate")

            tnew = np.arange(max_start_time, min_end_time, dt)

            data[f'{var}_{species}_{probe}']['Epoch'] = tnew
            data[f'{var}_{species}_{probe}']['Values'] = f(tnew)
