'''
Resample x to the resolution of y. (x and y must be pandas DataFrames)
'''
import numpy as np
from scipy.interpolate import interp1d

def resample(x, y):

    idx = y.index # Timestamps to resample to
    x = x[~x.index.duplicated()] # Check for any duplicate indices
    x = x.reindex(x.index.union(idx)).interpolate('linear').reindex(idx)

    return x

# def resample(x, y):

#     t = y.index # Times to resample to
#     vals = y.values

#     f = interp1d(t, vals, axis=0, kind='quadratic', fill_value='extrapolate')

#     dt = (y.index[1] - y.index[0]).total_seconds()

#     start_time = t[0].to_julian_date()
#     end_time = t[-1].to_julian_date()

#     tnew = np.arange(start_time, end_time, dt)
#     vals = f(tnew)

#     return vals