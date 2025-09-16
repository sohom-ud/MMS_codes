import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from src.utils.hdf_to_df import hdf_to_df
from src.utils.resample import resample

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

def plot_pdf(fname, out_fname, var, tscale, bins=50, color='blue', lw=1):

    with h5py.File(fname, 'r') as f:

        data = f[var]['Values'][tscale]

        #Normalize the data
        data -= np.nanmean(data)
        data /= np.nanstd(data)

        hist, bin_edges = np.histogram(data[~np.isnan(data)], bins, density=True)

        fig, axs = plt.subplots(1, 1, figsize=(6, 4))

        axs.plot(bin_edges[:-1], hist, color, lw)

        axs.set_xlabel(rf'{var} [nW/m$^3$]', fontsize=20)
        axs.set_ylabel('pdf', fontsize=20)

        plt.tight_layout()
        plt.savefig(out_fname, dpi=600)