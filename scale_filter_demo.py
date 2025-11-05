import os
import sys
from src.compute_routines.scale_filter_functions import *
from src.utils.write_scale_filtered import write_scale_filtered
import h5py
import time

#Uncomment the following line and add the correct path

#base_dir = PATH TO DATA DIRECTORY GOES HERE

data_dir = os.path.join(base_dir, 'scale_filtering')
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

gaussian_window_flag = 0

file = '20151209_050300_20151209_050400.h5' #Reconnection event is contained in this interval

fname = os.path.join(base_dir, file)

df_dict = hdf_to_df(fname)

max_timescale = 30000 #in milliseconds

step = 60
unit = 'ms'

#Get length of interval in milliseconds and round down to nearest integer
length_int = int((df_dict['v_spincorr_ion_1'].index[-1] - df_dict['v_spincorr_ion_1'].index[0]).total_seconds() * 1000)

trange = np.arange(step, max_timescale, step)

vars = ['PSi', 'PSe', 'Pi_uu_i', 'Pi_bb_i', 'Lambda_ub_i', 'Pi_uu_e', 'Pi_bb_e', 'Lambda_ub_e']
keys = ['Epochs', 'tscales', 'Values']
nested_list = ['Pi_bb_i', 'Pi_bb_e', 'Lambda_ub_i', 'Lambda_ub_e']

output_dict = {var: {} for var in vars}

for var in vars:
    output_dict[var]['tscales'] = trange
    if var in nested_list:
        output_dict[var]['Values'] = {probe: [] for probe in [1, 2, 3, 4]}
    else:
        output_dict[var]['Values'] = []

#Compute the scale filtered quantities at all time scales in the time range

for i, t in enumerate(trange):

    t = int(t)

    PiDi, pthi, PSi, Pi_uu_i, Pi_bb_i, Lambda_ub_i = compute_all_filtered(fname, t, unit, win_gauss=0, species='ion', reselectron=True)
    PiDe, pthe, PSe, Pi_uu_e, Pi_bb_e, Lambda_ub_e = compute_all_filtered(fname, t, unit, win_gauss=0, species='elc')

    output_dict['PSi']['Values'].append(PSi)
    output_dict['PiDi']['Values'].append(PiDi)
    output_dict['pthi']['Values'].append(pthi)
    output_dict['Pi_uu_i']['Values'].append(Pi_uu_i)
    
    output_dict['PSe']['Values'].append(PSe)
    output_dict['PiDe']['Values'].append(PiDe)
    output_dict['pthe']['Values'].append(pthe)
    output_dict['Pi_uu_e']['Values'].append(Pi_uu_e)
    
    for probe in [1, 2, 3, 4]:
        output_dict['Pi_bb_i']['Values'][probe].append(Pi_bb_i[probe])
        output_dict['Pi_bb_e']['Values'][probe].append(Pi_bb_e[probe])

        output_dict['Lambda_ub_i']['Values'][probe].append(Lambda_ub_i[probe])
        output_dict['Lambda_ub_e']['Values'][probe].append(Lambda_ub_e[probe])

    print(t)
    
write_scale_filtered(output_dict, file, data_dir)