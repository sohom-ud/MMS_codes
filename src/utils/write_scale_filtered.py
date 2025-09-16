'''
Write data to .h5 file
'''

import os
from pathlib import Path
import h5py
import pandas as pd

def write_scale_filtered(data, fname, data_dir):

    # Check whether file already exists or not
    out_fname = os.path.join(data_dir, rf'{os.path.splitext(fname)[0]}_filtered.h5')
    
    nested_list = ['Lambda_ub_i', 'Lambda_ub_e', 'Pi_bb_i', 'Pi_bb_e']

    if not os.path.exists(out_fname):
        with h5py.File(out_fname, 'a') as f:

            for key in data.keys():

                if key in nested_list:

                    # print(key)
                    f.create_group(key)
                    f.create_dataset(f"{key}/tscales", data=data[key]['tscales'])
                    for probe in [1, 2, 3, 4]:
                        f.create_dataset(f"{key}/Values/{probe}", data=data[key]['Values'][probe])
                else:
                        
                    # print(key)
                    f.create_group(key)
                    
                    f.create_dataset(f"{key}/tscales", data=data[key]['tscales'])
                    # f.create_dataset(f"{key}/Epochs", data=data[key]['Epochs'])
                    f.create_dataset(f"{key}/Values", data=data[key]['Values'])
                                
                print(f"Written {key} to file.")
    else:
        print("File already exists. Skipping to avoid overwrite.")
