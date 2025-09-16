import h5py

fname = r'Data/20151016_130600_20151016_130800.h5'

with h5py.File(fname, 'a') as f:

    for probe in [1, 2, 3, 4]:
        
        key = f'v_spincorr_{probe}_ion_reselectron'

        f.create_group(key)
        f.create_dataset(f"{key}/Epoch", data=f['N_elc_1']['Epoch'][...])
        f.create_dataset(f"{key}/Values", data=resample(f[f'v_spincorr_ion_{probe}']['Values'][...], f['N_elc_1']['Values'][...]))