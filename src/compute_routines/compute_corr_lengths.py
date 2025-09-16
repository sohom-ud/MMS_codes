import pandas as pd
import numpy as np
from src.utils.resample import resample
from src.utils.hdf_to_df import hdf_to_df

def corr_func(arr1,arr2,lag):

    l1 = len(arr1)
    l2 = len(arr2)
    data1 = arr1.iloc[0:l1-lag]
    data2 = arr2.iloc[lag:l2]
    mean1 = data1.mean()
    mean2 = data2.mean()
    corr = np.nanmean(data1.values * data2.values) - mean1 * mean2

    return corr

def compute_lambda_c(fname):

    df_dict = hdf_to_df(fname, vars=['b_gse_1', 'v_spincorr_ion_1'])

    B = resample(df_dict['b_gse_1'], df_dict['v_spincorr_ion_1'])
    v = df_dict['v_spincorr_ion_1']

    # for col in B.columns:
    #     B_A[col] = 

    vmag = np.nanmean(np.sqrt(v['x'] ** 2 + v['y'] ** 2 + v['z'] ** 2))

    dt = v.index[1] - v.index[0]

    max_lag = len(v) // 3

    lag_arr = list(range(0, max_lag))

    Bxx = np.array([corr_func(B['x'], B['x'], lag) for lag in lag_arr])
    Byy = np.array([corr_func(B['y'], B['y'], lag) for lag in lag_arr])
    Bzz = np.array([corr_func(B['z'], B['z'], lag) for lag in lag_arr])

    corr = (Bxx + Byy + Bzz) / 3.

    lambda_c = np.where(corr < np.exp(-1))[0][0] * dt.total_seconds() * vmag

    return lambda_c 