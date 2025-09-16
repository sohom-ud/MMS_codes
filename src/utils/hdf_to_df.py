'''
Converts .h5 files to pandas DataFrames
'''

import h5py
import numpy as np
import pandas as pd

def hdf_to_df(fname, vars="all"):

    data = h5py.File(fname, 'r')

    df_dict = dict()

    if vars=="all":

        varlist = list(data.keys())

    else:

        varlist = vars

    for key in varlist:

        x = data[key]['Epoch'][...]

        # x = [date.decode('utf-8') for date in x]
        
        y = data[key]['Values'][...]
        colnum = y.shape[1]

        if colnum == 1:
            cols = ['Epoch', 'val']  #Scalar variable
        elif colnum == 3 and len(y.shape) == 2:
            cols = ['Epoch', 'x', 'y', 'z']
        elif colnum == 4 and len(y.shape) == 2:
            cols = ['Epoch', 'x', 'y', 'z', 'mag']
        elif colnum == 9 or len(y.shape) == 3:
            y = data[key]['Values'][...].reshape(len(x), 9)
            cols = ['Epoch', 'xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']

        df_dict[key] = pd.DataFrame(data=np.column_stack([x, y]), columns=cols)
        # df_dict[key]['Epoch'] = list(map(lambda x: pd.Timestamp.utcfromtimestamp(x), df_dict[key]['Epoch']))
        df_dict[key]['Epoch'] = pd.to_datetime(df_dict[key]['Epoch'], unit='D', origin='julian')
        df_dict[key].set_index('Epoch', inplace=True)
        
        for col in df_dict[key].columns:
            df_dict[key][col] = pd.to_numeric(df_dict[key][col])

    data.close()

    return df_dict        

def hdf_to_df_filtered(fname, var="all"):

    nested_list = ['Lambda_ub_i', 'Lambda_ub_e', 'Pi_bb_i', 'Pi_bb_e']

    data = h5py.File(fname, 'r')

    df_dict = dict()

    if var == "all":
        varlist = list(data.keys())
    else:
        varlist = var

    cols = ['tscales', 'Values']

    for key in varlist:

        x = data[key]['tscales'][...]

        if key not in nested_list:
            y = data[key]['Values'][...]

            df_dict[key] = pd.DataFrame(np.column_stack([x, y]), columns=cols)

        else: 

            df_dict[key] = dict()
            df_dict[key]['Values'] = dict()
            for probe in [1, 2, 3, 4]:
                y = data[key]['Values'][str(probe)][...]
            
            df_dict[key]['Values'][probe] = pd.DataFrame(np.column_stack([x, y]), columns=cols)

    for col in df_dict[key].columns:
        df_dict[key][col] = pd.to_numeric(df_dict[key][col])

    data.close()

    return df_dict