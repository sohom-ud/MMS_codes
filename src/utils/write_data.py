'''
Write data to .h5 file
'''

import os
from pathlib import Path
import h5py
import pandas as pd

def write_data(data, trange, data_dir):

    start_time = pd.to_datetime(trange[0]).strftime('%Y%m%d_%H%M%S')
    end_time = pd.to_datetime(trange[1]).strftime('%Y%m%d_%H%M%S')

    fname = os.path.join(data_dir, f'{start_time}_{end_time}.h5')

    # Check whether file already exists or not
    if not os.path.exists(fname):
        with h5py.File(fname, 'a') as f:

            for key in data.keys():

                f.create_group(key)

                f.create_dataset(f"{key}/Epoch", data=data[key]['Epoch'])
                f.create_dataset(f"{key}/Values", data=data[key]['Values'])
                # data[key].to_hdf(fname, key=f'main/{key}')
                
                print(f"Written {key} to file.")
    else:
        print("File already exists. Skipping to avoid overwrite.")
