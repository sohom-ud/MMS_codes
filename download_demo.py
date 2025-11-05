from src import *
import numpy as np
import pandas as pd

if __name__ == "__main__":

    trange = ['2015-12-09/05:03:00', '2015-12-09/05:04:00'] # Wilder 2018 event 2

    # data_dir = PATH TO DATA DIRECTORY GOES HERE

    data = dict()
    
    for probe in [1, 2, 3, 4]:

        fgm_data = {}
        fpi_data_ions = {}
        fpi_data_electrons = {}
        edp_data = {}

        try:
            fgm_data = load_fgm(trange, probe=probe, wipe=False)
            data = dict(data, **fgm_data)
        except:
            print(f"No magnetic field data found for MMS{probe}.")

        try:
            fpi_data_ions = load_fpi_moms(trange, species='ion', probe=probe, wipe=False)
            data = dict(data, **fpi_data_ions)
        except:
            print(f"No ion moments found for MMS{probe}.")
        
        try:
            fpi_data_electrons = load_fpi_moms(trange, species='elc', probe=probe, wipe=False)
            data = dict(data, **fpi_data_electrons)
        except:
            print(f"No electron moments found for MMS{probe}.")

        try:
            edp_data = load_edp(trange, probe=probe, wipe=False)
            data = dict(data, **edp_data)
        except:
            print(f"No electric field data found for MMS{probe}.")

        data = dict(data, **edp_data)

    # Converting datetime objects to Julian day
    print("Converting datetimes to Julian day...")
    for key in data.keys():
        data[key]['Epoch'] = pd.DatetimeIndex(data[key]['Epoch']).to_julian_date()

    #Compute reciprocal vectors
    print("Computing reciprocal vectors at different resolutions...")
    k_B = compute_k(data) 
    k_vi = compute_k(data, res='vi')
    k_ve = compute_k(data, res='ve')

    data = dict(data, **k_B)    
    data = dict(data, **k_vi)
    data = dict(data, **k_ve)

   #Interpolate velocity, density, pressure so that all timestamps match
    print("Interpolating ion velocities, densities, pressures to common epochs...")
    interpolate_v_n_P(data, 'ion')

    print("Interpolating electron velocities, densities, pressures to common epochs...")
    interpolate_v_n_P(data, 'elc')

    print("Interpolating magnetic fields to common epochs...")
    interpolate_b(data)

    print("Interpolating electric fields to common epochs...")
    interpolate_E(data)

    print("Resampling ion velocities, densities and pressures to electron resolution...")
    resample_v_n_P(data)        

    print("Clipping time series to match start and end times for all spacecrafts")
    
    time_clip(data, trange)

    write_data(data, trange, data_dir)

    print(rf"Downloaded data for {trange[0]} - {trange[1]}")