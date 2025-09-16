import numpy as np
import pandas as pd
import h5py

from src.utils.compute_k import compute_k
from src.utils.interpolate_b import interpolate_b

mu0 = 4 * np.pi * 1e-7

def compute_jcurl(fname):

    data = h5py.File(fname, 'r+')

    k = dict()
    b = dict()
    bvals = dict()

    for probe in [1, 2, 3, 4]:
        
        k[probe] = data[f'k_{probe}']
        b[probe] = data[f'b_gse_{probe}']

    j = 0.0

    L = len(k[1]['Values'])

    for probe in [1, 2, 3, 4]:

        k[probe] = np.array(k[probe]['Values']).reshape(L, 3)
        bvals[probe] = b[probe]['Values'][:, :3]
    
        j += np.cross(k[probe], bvals[probe])

    j = (j * 1e-12 / mu0)/1e-9

    epoch = b[1]['Epoch'][...]

    epoch = [x.decode('utf-8') for x in epoch]

    j_df = pd.DataFrame(data=j, columns=['jx', 'jy', 'jz'])
    j_df.index = list(map(lambda x: pd.to_datetime(x), epoch))

    return j_df
