'''
Resample ion data to resolution of electrons
'''
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import interp1d

def resample_v_n_P(data):

    tnew = data['v_spincorr_elc_1']['Epoch'][...]

    for var in ['v_spincorr_ion', 'N_ion', 'Ptensor_ion', 'Temptensor_ion']:
        
        for probe in [1, 2, 3, 4]:

            t = data[f'{var}_{probe}']['Epoch']

            r = data[f'{var}_{probe}']['Values']

            df = pd.DataFrame(data=np.column_stack([t, r]))
            df.set_index(0, inplace=True)

            df = df.drop_duplicates()
            
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])

            t = df.index
            r = df.values

            f = interp1d(t, r, axis=0, kind='linear', fill_value="extrapolate")

            # Resampling positions to tnew

            data[f'{var}_{probe}_reselectron'] = dict()

            data[f'{var}_{probe}_reselectron']['Epoch'] = tnew
            data[f'{var}_{probe}_reselectron']['Values']= f(tnew)