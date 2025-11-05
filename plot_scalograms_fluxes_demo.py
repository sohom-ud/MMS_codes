import os
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from datetime import timedelta
from src.utils.hdf_to_df import hdf_to_df
from mpl_toolkits.axes_grid1 import make_axes_locatable

# base_dir = PATH TO DATA DIRECTORY GOES HERE

interval = r'20151209_050300_20151209_050400'

fname = os.path.join(base_dir, f'{interval}.h5')
filtered_fname = os.path.join(base_dir, 'scale_filtering', f'{interval}_filtered.h5')

filtered_data = h5py.File(filtered_fname, 'r')

df_dict = hdf_to_df(fname)

ni1 = df_dict['N_ion_1_reselectron']
vi1 = df_dict['v_spincorr_ion_1_reselectron']
B1 = df_dict['b_gse_1']
Ti1 = df_dict['Temptensor_ion_1_reselectron']
Ti = (Ti1['xx'] + Ti1['yy'] + Ti1['zz']) / 3.

epoch = ni1.index
# ni = np.nanmean(df_dict['N_ion_1'])
vi = np.sqrt(np.nanmean(vi1['x']**2 + vi1['y']**2 + vi1['z']**2))
B = np.sqrt(np.nanmean(B1['x']**2 + B1['y']**2 + B1['z']**2))

vi_loc = np.sqrt(vi1['x'] ** 2 + vi1['y'] ** 2 + vi1['z'] ** 2)

di_loc = 2.28e7/np.sqrt(ni1) * 1e-5 * 2 * np.pi
rhoi_loc = 1.02e2 * np.sqrt(Ti)/B * 2 * np.pi

Pi_uu_i = filtered_data['Pi_uu_i']['Values'][...] * 1e9
Pi_uu_e = filtered_data['Pi_uu_e']['Values'][...] * 1e9
Pi_bb_i = filtered_data['Pi_bb_i']['Values']['1'][...] * 1e9
Pi_bb_e = filtered_data['Pi_bb_e']['Values']['1'][...] * 1e9

Pi_uu_i_e = Pi_uu_i + Pi_uu_e
Pi_bb_i_e = Pi_bb_i + Pi_bb_e
Pi_uu_bb_i_e = Pi_uu_i + Pi_uu_e + Pi_bb_i + Pi_bb_e

start_time = datetime(2015, 12, 9, 5, 3, 56)
end_time = datetime(2015, 12, 9, 5, 3, 58)

start_time_idx = np.abs(epoch - start_time).argmin()
end_time_idx = np.abs(epoch - end_time).argmin()

tscales = filtered_data['Pi_uu_i']['tscales'][...]

fs = 150

plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
cm = 'RdBu_r'

fig, axs = plt.subplots(5, 1, figsize=(60, 70), sharex=True)

Pi_uu_i_im = axs[0].imshow(Pi_uu_i, origin='lower', cmap=cm, aspect='auto', 
           extent=[epoch[0], epoch[-1], tscales[0]/1000, tscales[-1]/1000],
           vmin=-0.1, vmax=0.1,
           interpolation='gaussian')

Pi_uu_e_im = axs[1].imshow(Pi_uu_e, origin='lower', cmap=cm, aspect='auto', 
           extent=[epoch[0], epoch[-1], tscales[0]/1000, tscales[-1]/1000],
           vmin=-3, vmax=3,
           interpolation='gaussian')

Pi_uu_i_e_im = axs[2].imshow(Pi_uu_i_e, origin='lower', cmap=cm, aspect='auto', 
           extent=[epoch[0], epoch[-1], tscales[0]/1000, tscales[-1]/1000],
           vmin=-3, vmax=3,
           interpolation='gaussian')

Pi_bb_i_e_im = axs[3].imshow(Pi_bb_i_e, origin='lower', cmap=cm, aspect='auto', 
           extent=[epoch[0], epoch[-1], tscales[0]/1000, tscales[-1]/1000],
           vmin=-0.1, vmax=0.1,
           interpolation='gaussian')

Pi_uu_bb_i_e_im = axs[4].imshow(Pi_uu_bb_i_e, origin='lower', cmap=cm, aspect='auto', 
           extent=[epoch[0], epoch[-1], tscales[0]/1000, tscales[-1]/1000],
           vmin=-3, vmax=3,
           interpolation='gaussian')


locator = mdates.MicrosecondLocator(interval=500000)
formatter = mdates.ConciseDateFormatter(locator)
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(formatter)

plt.xlim(start_time, end_time)

#------------- Adding colorbars -------------------------------------

cax_size = '2%'
cax_pad = 0.2
cax_x = 5
cax_y = 0.5

divider = make_axes_locatable(axs[0])
cax = divider.append_axes('right', size=cax_size, pad=cax_pad)
fig.colorbar(Pi_uu_i_im, cax=cax, orientation='vertical')
cax.text(cax_x, cax_y, r'$\Pi^{uu}_\mathrm{i}$', transform=cax.transAxes, va='center', rotation='horizontal', fontsize=fs)
cax.set_title('[nW/m$^3]$', fontsize=fs, y=1.1)

divider = make_axes_locatable(axs[1])
cax = divider.append_axes('right', size=cax_size, pad=cax_pad)
fig.colorbar(Pi_uu_e_im, cax=cax, orientation='vertical')
cax.text(cax_x, cax_y, r'$\Pi^{uu}_\mathrm{e}$', transform=cax.transAxes, va='center', rotation='horizontal', fontsize=fs)

divider = make_axes_locatable(axs[2])
cax = divider.append_axes('right', size=cax_size, pad=cax_pad)
fig.colorbar(Pi_uu_i_e_im, cax=cax, orientation='vertical')
# cax.text(cax_x, cax_y, r'$\sum\limits_\alpha \Pi^{uu}_\alpha$', transform=cax.transAxes, va='center', rotation='horizontal', fontsize=fs)
cax.text(cax_x, cax_y, r'$\Pi^{uu}_\textrm{total}$', transform=cax.transAxes, va='center', rotation='horizontal', fontsize=fs)

divider = make_axes_locatable(axs[3])
cax = divider.append_axes('right', size=cax_size, pad=cax_pad)
fig.colorbar(Pi_bb_i_e_im, cax=cax, orientation='vertical')
cax.text(cax_x, cax_y, r'$\Pi^{bb}_\textrm{total}$', transform=cax.transAxes, va='center', rotation='horizontal', fontsize=fs)

divider = make_axes_locatable(axs[4])
cax = divider.append_axes('right', size=cax_size, pad=cax_pad)
fig.colorbar(Pi_uu_bb_i_e_im, cax=cax, orientation='vertical')
cax.text(cax_x, cax_y, r'$\Pi$', transform=cax.transAxes, va='center', rotation='horizontal', fontsize=fs)

for ax in axs:
    ax.set_ylabel(r'$\tau$(s)', fontsize=fs)
    ax.set_ylim(bottom=0.06, top=50) #0.3s - 60s
    ax.set_yscale('log')
    ax.plot(di_loc.index, di_loc.values/vi, 'black', ls='--', lw=15)
    ax.plot(rhoi_loc.index, rhoi_loc.values/vi, 'black', lw=15)

    ax.text(start_time + timedelta(milliseconds=20), 0.5, r'$d_\mathrm{i}$', fontsize=fs*1.1)
    ax.text(start_time + timedelta(milliseconds=20), 7, r'$\rho_\mathrm{i}$', fontsize=fs*1.1)

x_annotate = -0.06
y_annotate = 1

axs[0].text(x_annotate, y_annotate, '(a)', fontsize=fs, transform=axs[0].transAxes)
axs[1].text(x_annotate, y_annotate, '(b)', fontsize=fs, transform=axs[1].transAxes)
axs[2].text(x_annotate, y_annotate, '(c)', fontsize=fs, transform=axs[2].transAxes)
axs[3].text(x_annotate, y_annotate, '(d)', fontsize=fs, transform=axs[3].transAxes)
axs[4].text(x_annotate, y_annotate, '(e)', fontsize=fs, transform=axs[4].transAxes)

axs[-1].set_xlabel('Datetime', fontsize=fs)
plt.subplots_adjust(hspace=0.1)

plt.tight_layout()

# plt.savefig(IMAGE PATH GOES HERE, dpi=50)
fig.close()