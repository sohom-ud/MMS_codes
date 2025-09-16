import pandas as pd
import numpy as np
from src.utils.hdf_to_df import hdf_to_df
from src.utils.resample import resample

mp = 1.67e-27
me = 9.1e-31
q = 1.6e-19

# Fluid energy transport term = \nabla\cdot(E_\alpha^f + \mathbf{P}_\alpha\cdot\mathbf{u}_\alpha)
def compute_fluid_transport(fname, species='ion', reselectron=True):

    df_dict = hdf_to_df(fname)

    k = dict()
    v = dict()

    for probe in [1, 2, 3, 4]:
        if species=='ion':
            if reselectron:
                # Resample the reciprocal vectors to the resolution of the species moments
                k[probe] = df_dict[f'k_{probe}_res_ve']
                v[probe] = df_dict[f'v_spincorr_{species}_{probe}_reselectron']
            else:
                # Resample the reciprocal vectors to the resolution of the species moments
                k[probe] = df_dict[f'k_{probe}_res_vi']
                v[probe] = df_dict[f'v_spincorr_{species}_{probe}']
    elif species=='elc':
        k[probe] = df_dict[f'k_{probe}_res_ve']
        v[probe] = df_dict[f'v_spincorr_{species}_{probe}']

    Ef = dict()
    Ef[probe] = 0.5 * m[species] * v[probe] ** 2 