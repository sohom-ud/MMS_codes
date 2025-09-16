from src import *
import numpy as np
import pandas as pd

if __name__ == "__main__":

    # trange = ['2016-12-09/09:03:04', '2016-12-09/09:04:00']  # Tai e-only reconnection interval
    # trange = ['2023-02-02/21:02:33', '2023-02-02/21:05:33']  # Unbiased turbulence campaign first orbit
    # trange = ['2017-12-26/06:12:43', '2017-12-26/06:52:23']  # 40 min long MSH interval
    # trange = ['2017-12-26/06:30:00', '2017-12-26/06:40:00']  # Subset of 40 min long MSH interval
    # trange = ['2017-12-26/06:31:00', '2017-12-26/06:36:00']  # Subset of 40 min long MSH interval
    # trange = ['2018-02-18/12:54:43', '2018-02-18/12:56:53'] # Reconnection interval (Manzini, PRL 2023)
    # trange = ['2017-07-11/22:34:00', '2017-07-11/22:34:04']
    # trange = ['2016-02-23/20:02:35', '2016-02-23/20:04:44'] # Manzini interval (2023b)
    # trange = ['2016-12-09/09:01:00', '2016-12-09/09:05:00']
    # trange = ['2023-02-24/03:01:24', '2023-02-24/03:03:53'] #Missing unbiased campaign interval
    # trange = ['2018-03-10/03:10:53', '2018-03-10/03:14:43']

    # trange = ['2017-09-28/06:31:33', '2017-09-28/07:01:43'] # 30 min long interval

    # trange = ['2016-02-23/20:02:04', '2016-02-23/20:05:40'] #Manzini interval full
    # trange = ['2017-01-28/05:32:30', '2017-01-28/05:33:00'] # Wilder 2018 event 1
    # trange = ['2015-12-09/05:03:54', '2015-12-09/05:04:00'] # Wilder 2018 event 2
    # trange = ['2015-11-04/04:35:00', '2015-11-04/04:36:00'] # Wilder 2018 event 3
    # trange = ['2017-01-01/14:17:00', '2017-01-01/14:17:30'] #Prayash new interval (e-only within ion jets)
    # trange = ['2015-10-16/13:06:00', '2015-10-16/13:08:00'] # Burch 2016 event
    # trange = ['2015-09-08/11:01:17', '2015-09-08/11:01:23'] # Eriksson 2016 event
    # trange = ['2015-12-08/11:20:40', '2015-12-08/11:20:50'] # Burch and Phan 2016 event
    
    # trange = ['2018-04-15/04:32:15', '2018-04-15/04:33:45'] # Burch et. al. 2020 event
    # trange = ['2015-09-08/11:00:04', '2015-09-08/11:02:00'] # Eriksson et. al. 2016

    # trange = ['2015-10-02/10:58:44', '2015-10-02/11:01:40']  # Alfven vortex interval 1 (Wang 2019)
    # trange = ['2017-11-02/04:26:23', '2017-11-02/04:27:30'] # Shock crossing (Agapitov 2023)
    # trange = ['2016-02-23/20:02:35', '2016-02-23/20:04:44']
    # trange = ['2016-02-23/20:02:04', '2016-02-23/20:05:40']

    # trange = ['2017-07-11/22:30:00', '2017-07-11/22:40:00'] # Torbert 2018 event

    # trange = ['2015-12-09/05:03:00', '2015-12-09/05:04:00'] # Wilder 2018 event 2

    # trange = ['2024-05-10/18:40:45', '2024-05-11/14:48:57'] # CME event (May 2024)

    # trange = ['2015-10-07/12:06:55', '2015-10-07/12:08:00'] # Bow shock event (Lei 2024)
    # trange = ['2017-01-28/05:32:30', '2017-01-28/05:32:35']

    # trange = ['2018-03-21/21:35:50', '2018-03-21/21:36:30'] # Quasi-parallel shock (Yao 2024)
    # trange = ['2023-04-24/04:00:13', '2023-04-24/04:03:30']

    # trange = ['2017-01-20/03:03:53', '2017-01-20/03:06:43']
    # trange = ['2017-01-20/03:28:53', '2017-01-20/03:32:03']

    # trange = ['2018-04-17/18:38:43', '2018-04-17/18:45:43']
    # trange = ['2023-02-02/21:20:33', '2023-02-02/21:23:33'] #Ciccio test event

    # trange = ['2017-06-19/03:57:35.0', '2017-06-19/03:57:41'] #DF event
    trange = ['2017-06-24/23:58:22', '2017-06-24/23:58:37']

    # interval_list = pd.read_csv(r'/home/sohom/MMS_PySPEDAS/interval_list.txt', delim_whitespace=True, header=None, names=['start_time', 'end_time', 'duration'])    

    data_dir = r'/home/sroy/Documents/IWF_research/Codes'

    data = dict()
    
    for probe in [1, 2, 3, 4]:

        fgm_data = {}
        fpi_data_ions = {}
        fpi_data_electrons = {}
        edp_data = {}

        try:
            fgm_data = load_fgm(trange, probe=probe, wipe=False)
            data = dict(data, **fgm_data)
            # fgm_data = time_clip(fgm_data, trange)
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
        # data[key]['Epoch'] = np.array([pd.Timestamp(x).to_julian_date() for x in data[key]['Epoch']])
        data[key]['Epoch'] = pd.DatetimeIndex(data[key]['Epoch']).to_julian_date()

    #Compute reciprocal vectors
    print("Computing reciprocal vectors at different resolutions...")
    k_B = compute_k(data) 
    k_vi = compute_k(data, res='vi')
    k_ve = compute_k(data, res='ve')

    data = dict(data, **k_B)    
    data = dict(data, **k_vi)
    data = dict(data, **k_ve)
    # # Interpolate velocity, density, pressure so that all timestamps match
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

    # fgm_data = time_clip(fgm_data, trange)
    # k = time_clip(k, trange)
    
    print("Clipping time series to match start and end times for all spacecrafts")
    
    time_clip(data, trange)

    write_data(data, trange, data_dir)

    print(rf"Downloaded data for {trange[0]} - {trange[1]}")