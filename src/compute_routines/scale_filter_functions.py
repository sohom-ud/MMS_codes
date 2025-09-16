import pandas as pd
import numpy as np
from datetime import timedelta
from src.compute_routines.compute_PiD_functions import *
from src.utils.methods import *
from src.utils.resample import resample
from src.utils.hdf_to_df import hdf_to_df

mi = 1.67e-27 # Mass of proton in kg
c = 3e8 # Speed of light in m/s

def filt(x, t, unit='S', win_gauss=0, species='ion'): # t is the time scale (window) in seconds over which x is being filtered; x is considered to be a pandas Series (or DataFrame) here

    if species == 'ion':
        resolution = 150
    elif species == 'elc':
        resolution = 30

    if not win_gauss:
        x_filtered = x.rolling(window=f'{t}{unit}', center=True).mean()
    else:
        x_filtered = x.rolling(window=f'{t}{unit}', win_type='gaussian', center=True).mean(std=(t/resolution)**2/6)

    return x_filtered

def favre_filt(x, t, n, unit='S', win_gauss=0, species='ion'): 

    xf = x.values * n.values
    xf = pd.DataFrame(xf, index=x.index)

    if species=='ion':
        resolution = 150
    elif species=='elc':
        resolution = 30

    if not win_gauss:
        x_favre_filtered = filt(xf, t, unit, 0, species).values/filt(n, t, unit, 0, species).values
    else:
        x_favre_filtered = filt(xf, t, unit, 1, species).values/filt(n, t, unit, 1, species).values

    x_favre_filtered = pd.DataFrame(x_favre_filtered, index=x.index, columns=x.columns)

    return x_favre_filtered

def compute_gradv_filtered(fname, t, unit, win_gauss=0, species='ion', reselectron=True, favre=1):

    ## gradv_filtered is divj

    df_dict = hdf_to_df(fname)
        
    v = dict()
    k = dict()
    n = dict()

    for probe in [1, 2, 3, 4]:

        if species=='ion':
            if reselectron:
                v[probe] = df_dict[rf'v_spincorr_{species}_{probe}_reselectron']
                k[probe] = df_dict[rf'k_{probe}_res_ve']
                n[probe] = df_dict[rf'N_{species}_{probe}_reselectron']
            else:
                v[probe] = df_dict[rf'v_spincorr_{species}_{probe}']
                k[probe] = df_dict[rf'k_{probe}_res_vi']
                n[probe] = df_dict[rf'N_{species}_{probe}']
        else:
            v[probe] = df_dict[rf'v_spincorr_{species}_{probe}']
            k[probe] = df_dict[rf'k_{probe}_res_ve']
            n[probe] = resample(df_dict[rf'N_{species}_{probe}'], v[probe])

    v_filt = dict()
    for probe in [1, 2, 3, 4]:
        if favre:
            v_filt[probe] = favre_filt(v[probe], t, n[probe], unit, win_gauss, species)
        else:
            v_filt[probe] = filt(v[probe], t, unit, win_gauss, species)
   
    gradv_filtered = pd.DataFrame()
    gradv_filtered.index = k[1].index

    for comp1 in ['x', 'y', 'z']:

        for comp2 in ['x', 'y', 'z']:

            gradv_filtered[f'd{comp1}v{comp2}'] = 0.0

            for probe in [1, 2, 3, 4]:

                gradv_filtered[f'd{comp1}v{comp2}'] += k[probe][comp1] * v_filt[probe][comp2]

    return gradv_filtered

def compute_gradB_filtered(fname, t, unit, win_gauss=0, reselectron=True, favre=0):

    # gradB_filtered is diBj

    df_dict = hdf_to_df(fname)

    v = dict()        
    B = dict()
    k = dict()
    n = dict()

    for probe in [1, 2, 3, 4]:

        v[probe] = df_dict[rf'v_spincorr_elc_{probe}']
        n[probe] = resample(df_dict[rf'N_elc_{probe}'], v[probe])
        B[probe] = resample(df_dict[rf'b_gse_{probe}'], v[probe])
        k[probe] = df_dict[rf'k_{probe}_res_ve']
        B[probe] = B[probe].drop('mag', axis=1)

    B_filt = dict()
    for probe in [1, 2, 3, 4]:

        if favre:
            B_filt[probe] = favre_filt(B[probe], t, n[probe], unit, win_gauss, 'elc')
        else:
            B_filt[probe] = filt(B[probe], t, unit, win_gauss, 'elc')
   
    gradB_filtered = pd.DataFrame()
    gradB_filtered.index = k[1].index

    for comp1 in ['x', 'y', 'z']:

        for comp2 in ['x', 'y', 'z']:

            gradB_filtered[f'd{comp1}B{comp2}'] = 0.0

            for probe in [1, 2, 3, 4]:

                gradB_filtered[f'd{comp1}B{comp2}'] += k[probe][comp1] * B_filt[probe][comp2]

    return gradB_filtered

def compute_Omega_filtered(fname, t, unit, species, win_gauss=0, reselectron=True, favre=0):

    #Omega_filtered = 0.5 * (divj - djvi)

    gradv_filtered = compute_gradv_filtered(fname, t, unit, win_gauss, species, reselectron)

    Omega_filtered = pd.DataFrame()
    Omega_filtered.index = gradv_filtered.index

    for comp1 in ['x', 'y', 'z']:
        for comp2 in ['x', 'y', 'z']:
            Omega_filtered[f'{comp1}{comp2}'] = 0.5 * (gradv_filtered[f'd{comp1}v{comp2}'] - gradv_filtered[f'd{comp2}v{comp1}'])

    return Omega_filtered

def compute_Omega_B_filtered(fname, t, unit, species, win_gauss=0, reselectron=True):

    #Omega_B_filtered = 0.5 * (diBj - djBi)

    gradB_filtered = compute_gradB_filtered(fname, t, unit, win_gauss, reselectron, favre=0)

    Omega_B_filtered = pd.DataFrame()
    Omega_B_filtered.index = gradB_filtered.index

    for comp1 in ['x', 'y', 'z']:
        for comp2 in ['x', 'y', 'z']:
            Omega_B_filtered[f'{comp1}{comp2}'] = 0.5 * (gradB_filtered[f'd{comp1}B{comp2}'] - gradB_filtered[f'd{comp2}B{comp1}'])

    return Omega_B_filtered

def compute_P_filtered(fname, t, unit, win_gauss=0, species='ion', reselectron=True):

    df_dict = hdf_to_df(fname)

    P = compute_avg_P(fname, species, reselectron)

    P_filtered = filt(P, t, unit, win_gauss, species)

    return P_filtered

def compute_baropycnal_work(fname, t, unit, win_gauss=0, species='ion', reselectron=True):

    df_dict = hdf_to_df(fname)

    v = dict()
    k = dict()
    n = dict()
    P = dict()

    for probe in [1, 2, 3, 4]:

        if species=='ion':
            if reselectron:
                v[probe] = df_dict[rf'v_spincorr_{species}_{probe}_reselectron']
                k[probe] = df_dict[rf'k_{probe}_res_ve']
                n[probe] = df_dict[rf'N_{species}_{probe}_reselectron']
                P[probe] = df_dict[rf'Ptensor_{species}_{probe}_reselectron']
            else:
                v[probe] = df_dict[rf'v_spincorr_{species}_{probe}']
                k[probe] = df_dict[rf'k_{probe}_res_vi']
                n[probe] = df_dict[rf'N_{species}_{probe}']
                P[probe] = df_dict[rf'Ptensor_{species}_{probe}']
        else:
            v[probe] = df_dict[rf'v_spincorr_{species}_{probe}']
            k[probe] = df_dict[rf'k_{probe}_res_ve']
            n[probe] = resample(df_dict[rf'N_{species}_{probe}'], v[probe])
            P[probe] = df_dict[rf'Ptensor_{species}_{probe}']

    v_favre_filt = dict()
    v_filt = dict()

    for probe in [1, 2, 3, 4]:
        v_favre_filt[probe] = favre_filt(v[probe], t, n[probe], unit, win_gauss, species)
        v_filt[probe] = filt(v[probe], t, unit, win_gauss, species)
    
    P_filtered = dict()

    for probe in [1, 2, 3, 4]:
        P_filtered[probe] = filt(P[probe], t, unit, win_gauss, species)

    div_P_filtered = pd.DataFrame()
    div_P_filtered.index = k[1].index

    for comp1 in ['x', 'y', 'z']:
        for comp2 in ['x', 'y', 'z']:
            div_P_filtered[comp1] = 0.0
            for probe in [1, 2, 3, 4]:
                div_P_filtered[comp1] += k[probe][comp2] * P[probe][f'{comp1}{comp2}']

    baro_work = 0.0
    for comp in ['x', 'y', 'z']:
        baro_work += div_P_filtered[comp] * (v_favre_filt[1][comp] - v_filt[1][comp])

    return baro_work

def compute_PS_filtered(fname, t, unit, win_gauss=0, species='ion', reselectron=True):

    df_dict = hdf_to_df(fname)
        
    P_filtered = compute_P_filtered(fname, t, unit, win_gauss, species, reselectron)

    S_filtered = compute_gradv_filtered(fname, t, unit, win_gauss, species, reselectron)

    PS_filtered = pd.DataFrame()
    PS_filtered.index = P_filtered.index
    PS_filtered[f'PS_{species}'] = 0.0

    for comp1 in ['x', 'y', 'z']:
        for comp2 in ['x', 'y', 'z']:
            PS_filtered[f'PS_{species}'] += P_filtered[f'{comp1}{comp2}'] * S_filtered[f'd{comp1}v{comp2}']

    return PS_filtered

def compute_ptheta_filtered(fname, t, unit, win_gauss=0, species='ion', reselectron=True):

    df_dict = hdf_to_df(fname)
        
    p = compute_p(fname, species, reselectron)

    p_filtered = filt(p, t, unit, win_gauss, species, reselectron)
    
    S_filtered = compute_gradv_filtered(fname, t, unit, win_gauss, species, reselectron)

    theta_filtered = (S_filtered['dxvx'] + S_filtered['dyvy'] + S_filtered['dzvz'])

    ptheta_filtered = pd.DataFrame()

    ptheta_filtered.index = p_filtered.index

    ptheta_filtered[f'ptheta_{species}'] = p_filtered.values * theta_filtered.values

    return ptheta_filtered

def compute_PiD_filtered(fname, t, unit, win_gauss=0, species='ion', reselectron=True):
    
    Pi = compute_Pi_ij(fname, species, reselectron)
    
    Pi_filtered = filt(Pi, t, unit, win_gauss, species, reselectron)
    
    df_dict = hdf_to_df(fname)
            
    S_filtered = compute_gradv_filtered(fname, t, unit, species, reselectron)

    theta_filtered = (S_filtered['dxvx'] + S_filtered['dyvy'] + S_filtered['dzvz'])

    D_filtered = pd.DataFrame()
    D_filtered.index = S_filtered.index

    for comp1 in ['x', 'y', 'z']:
        for comp2 in ['x', 'y', 'z']:
            D_filtered[f'{comp1}{comp2}'] = (S_filtered[f'd{comp1}v{comp2}'] + S_filtered[f'd{comp2}v{comp1}'])/2.0 - (theta_filtered * (comp1 == comp2))/3.0

    PiD_filtered = pd.DataFrame()

    PiD_filtered.index = Pi_filtered.index
    PiD_filtered[f'PiD_{species}'] = 0.0

    for comp1 in ['x', 'y', 'z']:

        for comp2 in ['x', 'y', 'z']:

            PiD_filtered[f'PiD_{species}'] += Pi_filtered[f'{comp1}{comp2}'] * D_filtered[f'{comp1}{comp2}']

    return PiD_filtered

def compute_tau_u(fname, probe, t, unit, win_gauss=0, species='ion', reselectron=True):
    
    df_dict = hdf_to_df(fname)
        
    v = dict()
    n = dict()

    if species=='ion' and reselectron:
        v[probe] = df_dict[rf'v_spincorr_{species}_{probe}_reselectron']
        n[probe] = df_dict[rf'N_{species}_{probe}_reselectron']
    else:
        v[probe] = df_dict[rf'v_spincorr_{species}_{probe}']
        n[probe] = resample(df_dict[rf'N_{species}_{probe}'], v[probe])

    tau_u = favre_filt(mult(v[probe], v[probe]), t, n[probe], unit, win_gauss, species) - mult(favre_filt(v[probe], t, n[probe], unit, win_gauss, species), favre_filt(v[probe], t, n[probe], unit, win_gauss, species))
    
    return tau_u

def compute_tau_b(fname, probe, t, unit, win_gauss=0, species='ion', reselectron=True):

    df_dict = hdf_to_df(fname)
        
    v = dict()
    B = dict()
    n = dict()

    if species=='ion' and reselectron:
        v[probe] = df_dict[rf'v_spincorr_{species}_{probe}_reselectron']
        B[probe] = resample(df_dict[rf'b_gse_{probe}'], v[probe])
        n[probe] = df_dict[rf'N_{species}_{probe}_reselectron']
    else:
        v[probe] = df_dict[rf'v_spincorr_{species}_{probe}']
        B[probe] = resample(df_dict[rf'b_gse_{probe}'], v[probe])
        n[probe] = resample(df_dict[rf'N_{species}_{probe}'], v[probe])


    B[probe] = B[probe].drop('mag', axis=1)

    tau_b = favre_filt(cross(v[probe], B[probe]), t, n[probe], unit, win_gauss, species) - cross(favre_filt(v[probe], t, n[probe], unit, win_gauss, species), favre_filt(B[probe], t, n[probe], unit, win_gauss, species))

    return tau_b

def compute_tau_e(fname, probe, t, unit, win_gauss=0, species='ion', reselectron=True):

    df_dict = hdf_to_df(fname)
    
    v = dict()
    E = dict()
    n = dict()

    if species=='ion' and reselectron:
        v[probe] = df_dict[rf'v_spincorr_{species}_{probe}_reselectron']
        n[probe] = df_dict[rf'N_{species}_{probe}_reselectron']
        E[probe] = resample(df_dict[rf'edp_dce_gse_{probe}'], v[probe])
    else:
        v[probe] = df_dict[rf'v_spincorr_{species}_{probe}']
        n[probe] = resample(df_dict[rf'N_{species}_{probe}'], v[probe])
        E[probe] = resample(df_dict[rf'edp_dce_gse_{probe}'], v[probe])

    E_favre_filt = favre_filt(E[probe], t, n[probe], unit, win_gauss, species)
    E_filt = filt(E[probe], t, unit, win_gauss, species)

    tau_e = E_favre_filt - E_filt

    return tau_e

def compute_Pi_uu(fname, t, unit, win_gauss=0, species='ion', reselectron=True):

    df_dict = hdf_to_df(fname)
        
    v = dict()
    k = dict()
    n = dict()
    B = dict()

    if species=='ion' and reselectron:
        for probe in [1, 2, 3, 4]:
            v[probe] = df_dict[rf'v_spincorr_{species}_{probe}_reselectron']
            # k[probe] = df_dict[rf'k_{probe}_reselectron']
            k[probe] = df_dict[f'k_{probe}_res_ve']
            n[probe] = df_dict[rf'N_{species}_{probe}_reselectron']
            B[probe] = resample(df_dict[rf'b_gse_{probe}'], v[probe])
    else:
        for probe in [1, 2, 3, 4]:
            v[probe] = df_dict[rf'v_spincorr_{species}_{probe}']
            # k[probe] = resample(df_dict[rf'k_{probe}'], v[probe])
            if species=='elc': 
                k[probe] = df_dict[f'k_{probe}_res_ve']
            else:
                k[probe] = df_dict[f'k_{probe}_res_vi']
            n[probe] = resample(df_dict[rf'N_{species}_{probe}'], v[probe])
            B[probe] = resample(df_dict[rf'b_gse_{probe}'], v[probe])

    if species == 'ion':
        q = 1.6e-19
        m = 1.67e-27
    elif species == 'elc':
        q = -1.6e-19
        m = 9.1e-31

    rho_filt = dict()
    n_filt = dict()
    tau_u = dict()
    tau_b = dict()
    u = dict()

    for probe in [1, 2, 3, 4]:

        rho_filt[probe] = filt(m * n[probe], t, unit, win_gauss, species)
        n_filt[probe] = filt(n[probe], t, unit, win_gauss, species)
        tau_u[probe] = compute_tau_u(fname, probe, t, unit, win_gauss, species, reselectron)

        u[probe] = favre_filt(v[probe], t, n[probe], unit, win_gauss, species)

        tau_b[probe] = compute_tau_b(fname, probe, t, unit, win_gauss, species, reselectron)

    rho_filt_avg = (rho_filt[1] + rho_filt[2] + rho_filt[3] + rho_filt[4]) / 4.
    tau_u_avg = (tau_u[1] + tau_u[2] + tau_u[3] + tau_u[4]) / 4.

    Pi_uu_T1 = - dot(rho_filt_avg.values * tau_u_avg, grad(k, u)) * 1e6 * 1e6 # W/m^3

    Pi_uu_T2 = 0.0
    
    for probe in [1, 2, 3, 4]:
        
        Pi_uu_T2 += - q * n_filt[probe].values * dot(tau_b[probe], u[probe]) * 1e6 * 1e3 * 1e-9 * 1e3 # W/m^3

    Pi_uu_T2 /= 4.0

    Pi_uu = Pi_uu_T1 + Pi_uu_T2

    return Pi_uu # W/m^3

def compute_Lambda_ub(fname, probe, t, unit, win_gauss=0, species='ion', reselectron=True):

    df_dict = hdf_to_df(fname)
        
    v = dict()
    n = dict()
    E = dict()

    if species=='ion' and reselectron:
        for probe in [1, 2, 3, 4]:
            v[probe] = df_dict[rf'v_spincorr_{species}_{probe}_reselectron']
            n[probe] = df_dict[rf'N_{species}_{probe}_reselectron']
            E[probe] = resample(df_dict[rf'edp_dce_gse_{probe}'], v[probe])
    else:
        for probe in [1, 2, 3, 4]:
            v[probe] = df_dict[rf'v_spincorr_{species}_{probe}']
            n[probe] = resample(df_dict[rf'N_{species}_{probe}'], v[probe])
            E[probe] = resample(df_dict[rf'edp_dce_gse_{probe}'], v[probe])

    if species == 'ion':
        q = 1.6e-19
    elif species == 'elc':
        q = -1.6e-19

    n_filt = filt(n[probe], t, unit, win_gauss, species) # /cm^3
    
    E_filt = favre_filt(E[probe], t, n[probe], unit, win_gauss, species) # mV/m

    v_favre_filt = favre_filt(v[probe], t, n[probe], unit, win_gauss, species) # km/s

    Lambda_ub = - q * n_filt.values * dot(E_filt, v_favre_filt) * 1e6 * 1e9

    return Lambda_ub # in nW/m^3

def compute_Pi_bb(fname, probe, t, unit, win_gauss=0, species='ion', reselectron=True):

    df_dict = hdf_to_df(fname)
        
    v = dict()
    n = dict()
    E = dict()

    if species=='ion' and reselectron:
        for probe in [1, 2, 3, 4]:
            v[probe] = df_dict[rf'v_spincorr_{species}_{probe}_reselectron']
            n[probe] = df_dict[rf'N_{species}_{probe}_reselectron']
            E[probe] = resample(df_dict[rf'edp_dce_gse_{probe}'], v[probe])
    else:
        for probe in [1, 2, 3, 4]:
            v[probe] = df_dict[rf'v_spincorr_{species}_{probe}']
            n[probe] = resample(df_dict[rf'N_{species}_{probe}'], v[probe])
            E[probe] = resample(df_dict[rf'edp_dce_gse_{probe}'], v[probe])

    if species == 'ion':
        q = 1.6e-19
    elif species == 'elc':
        q = -1.6e-19

    n_filt = filt(n[probe], t, unit, win_gauss, species) # /cm^3

    tau_e = compute_tau_e(fname, probe, t, unit, win_gauss, species, reselectron) # mV/m

    v_favre_filt = favre_filt(v[probe], t, n[probe], unit, win_gauss, species) # km/s

    Pi_bb = - q * n_filt.values * dot(tau_e, v_favre_filt) * 1e6 # W/m^3

    return Pi_bb # W/m^3

def compute_all_filtered(fname, t, unit, win_gauss=0, species='ion', reselectron=True):

    if species=='ion':
        m = 1.67e-27
    elif species=='elc':
        m = 9.1e-31

    df_dict = hdf_to_df(fname)

    if species=='ion':
        if reselectron:
            v = {probe: df_dict[rf'v_spincorr_{species}_{probe}_reselectron'] for probe in [1, 2, 3, 4]}
            k = {probe: df_dict[rf'k_{probe}_res_ve'] for probe in [1, 2, 3, 4]}
            n = {probe: df_dict[rf'N_{species}_{probe}_reselectron'] for probe in [1, 2, 3, 4]}
            B = {probe: resample(df_dict[rf'b_gse_{probe}'].drop('mag', axis=1), v[probe]) for probe in [1, 2, 3, 4]}
            E = {probe:resample(df_dict[rf'edp_dce_gse_{probe}'], v[probe]) for probe in [1, 2, 3, 4]}
        else:
            v = {probe: df_dict[rf'v_spincorr_{species}_{probe}'] for probe in [1, 2, 3, 4]}
            k = {probe: df_dict[rf'k_{probe}_res_vi'] for probe in [1, 2, 3, 4]}
            n = {probe: df_dict[rf'N_{species}_{probe}'] for probe in [1, 2, 3, 4]}
            B = {probe: resample(df_dict[rf'b_gse_{probe}'].drop('mag', axis=1), v[probe]) for probe in [1, 2, 3, 4]}
            E = {probe:resample(df_dict[rf'edp_dce_gse_{probe}'], v[probe]) for probe in [1, 2, 3, 4]}
    else:
        v = {probe: df_dict[rf'v_spincorr_{species}_{probe}'] for probe in [1, 2, 3, 4]}
        k = {probe: df_dict[rf'k_{probe}_res_ve'] for probe in [1, 2, 3, 4]}
        n = {probe: resample(df_dict[rf'N_{species}_{probe}'], v[probe]) for probe in [1, 2, 3, 4]}
        B = {probe: resample(df_dict[rf'b_gse_{probe}'].drop('mag', axis=1), v[probe]) for probe in [1, 2, 3, 4]}
        E = {probe:resample(df_dict[rf'edp_dce_gse_{probe}'], v[probe]) for probe in [1, 2, 3, 4]}

    v_favre_filt = {probe: favre_filt(v[probe], t, n[probe], unit, win_gauss, species) for probe in [1, 2, 3, 4]}

    rho_filt = {probe: filt(m * n[probe], t, unit, win_gauss, species) for probe in [1, 2, 3, 4]}
    n_filt = {probe: filt(n[probe], t, unit, win_gauss, species) for probe in [1, 2, 3, 4]}

    # P_filtered = compute_P_filtered(fname, t, unit, win_gauss, species, reselectron)
    
    # Pi = compute_Pi_ij(fname, species, reselectron)
    
    # Pi_filtered = filt(Pi, t, unit, win_gauss, species)
       
    # S_filtered = compute_S_filtered(fname, t, unit, win_gauss, species, reselectron)

    # theta_filtered = (S_filtered['dxvx'] + S_filtered['dyvy'] + S_filtered['dzvz'])

    # D_filtered = pd.DataFrame()
    # D_filtered.index = S_filtered.index

    # p = compute_p(fname, species, reselectron)

    # p_filtered = filt(p, t, unit, win_gauss, species)
    
    # theta_filtered = (S_filtered['dxvx'] + S_filtered['dyvy'] + S_filtered['dzvz'])

    # ptheta_filtered = pd.DataFrame()
    # ptheta_filtered.index = p_filtered.index

    # ptheta_filtered[f'ptheta_{species}'] = p_filtered.values * theta_filtered.values

    # PS_filtered = pd.DataFrame()
    # PS_filtered.index = P_filtered.index
    # PS_filtered[f'PS_{species}'] = 0.0
    
    # for comp1 in ['x', 'y', 'z']:
    #     for comp2 in ['x', 'y', 'z']:
    #         D_filtered[f'{comp1}{comp2}'] = (S_filtered[f'd{comp1}v{comp2}'] + S_filtered[f'd{comp2}v{comp1}'])/2.0 - (theta_filtered * (comp1 == comp2))/3.0

    # PiD_filtered = pd.DataFrame()

    # PiD_filtered.index = Pi_filtered.index
    # PiD_filtered[f'PiD_{species}'] = 0.0

    # for comp1 in ['x', 'y', 'z']:
    #     for comp2 in ['x', 'y', 'z']:
    #         PiD_filtered[f'PiD_{species}'] += Pi_filtered[f'{comp1}{comp2}'] * D_filtered[f'{comp1}{comp2}']
    #         PS_filtered[f'PS_{species}'] += P_filtered[f'{comp1}{comp2}'] * S_filtered[f'd{comp1}v{comp2}']

    tau_u = {probe: favre_filt(mult(v[probe], v[probe]), t, n[probe], unit, win_gauss, species) - mult(favre_filt(v[probe], t, n[probe], unit, win_gauss, species), favre_filt(v[probe], t, n[probe], unit, win_gauss, species)) for probe in [1, 2, 3, 4]}
    tau_b = {probe: favre_filt(cross(v[probe], B[probe]), t, n[probe], unit, win_gauss, species) - cross(favre_filt(v[probe], t, n[probe], unit, win_gauss, species), favre_filt(B[probe], t, n[probe], unit, win_gauss, species)) for probe in [1, 2, 3, 4]}
    tau_e = {probe: favre_filt(E[probe], t, n[probe], unit, win_gauss, species) - filt(E[probe], t, unit, win_gauss, species) for probe in [1, 2, 3, 4]}

    if species == 'ion':
        q = 1.6e-19
    elif species == 'elc':
        q = -1.6e-19

    rho_filt_avg = (rho_filt[1] + rho_filt[2] + rho_filt[3] + rho_filt[4]) / 4.
    tau_u_avg = (tau_u[1] + tau_u[2] + tau_u[3] + tau_u[4]) / 4.

    Pi_uu_T1 = - dot(rho_filt_avg.values * tau_u_avg, grad(k, v_favre_filt)) * 1e6 * 1e6

    Pi_uu_T2 = 0.0
    
    for probe in [1, 2, 3, 4]:
        
        Pi_uu_T2 += - q * n_filt[probe].values * dot(tau_b[probe], v_favre_filt[probe]) * 1e6 * 1e3 * 1e-9 * 1e3

    Pi_uu_T2 /= 4.0

    Pi_uu = Pi_uu_T1 + Pi_uu_T2
    
    Lambda_ub = {probe: - q * n_filt[probe].values * dot(favre_filt(E[probe], t, n[probe], unit, win_gauss, species), v_favre_filt[probe]) * 1e6 * 1e9 for probe in [1, 2, 3, 4]}

    Pi_bb = {probe: - q * n_filt[probe].values * dot(tau_e[probe], v_favre_filt[probe]) * 1e6 for probe in [1, 2, 3, 4]}

    # return PiD_filtered, ptheta_filtered, PS_filtered, Pi_uu, Pi_bb, Lambda_ub
    return Pi_uu, Pi_bb, Lambda_ub

def calc_trace_S3_filtered(fname, t, unit, win_gauss=0, species='ion', reselectron=True):

    gradv_filt = compute_gradv_filtered(fname, t, unit, win_gauss, species, reselectron)

    S_filt = pd.DataFrame()
    S_filt.index = gradv_filt.index

    for comp1 in ['x', 'y', 'z']:
        for comp2 in ['x', 'y', 'z']:
            S_filt[f'{comp1}{comp2}'] = 0.5 * (gradv_filt[f'd{comp1}v{comp2}'] + gradv_filt[f'd{comp2}v{comp1}'])

    tr_S3 = (S_filt['xx'] ** 3 + S_filt['yy'] ** 3 + S_filt['zz'] ** 3) + \
             3 * (S_filt['xx']*S_filt['xy']**2 + S_filt['xx']*S_filt['xz']**2 + S_filt['yy']*S_filt['xy']**2 + \
                  S_filt['zz']*S_filt['xz']**2 + S_filt['yy']*S_filt['yz']**2 + S_filt['zz']*S_filt['yz']**2) + \
             6 * S_filt['xy']*S_filt['yz']*S_filt['xz']

    return tr_S3

def calc_trace_S3_B_filtered(fname, t, unit, win_gauss=0, reselectron=True):

    gradB_filt = compute_gradB_filtered(fname, t, unit, win_gauss, reselectron)

    S_B_filt = pd.DataFrame()
    S_B_filt = gradB_filt.index

    for comp1 in ['x', 'y', 'z']:
        for comp2 in ['x', 'y', 'z']:
            S_B_filt[f'{comp1}{comp2}'] = 0.5 * (gradB_filt[f'd{comp1}B{comp2}'] + gradB_filt[f'd{comp2}B{comp1}'])

    tr_S3 = (S_B_filt['xx'] ** 3 + S_B_filt['yy'] ** 3 + S_B_filt['zz'] ** 3) + \
             3 * (S_B_filt['xx']*S_B_filt['xy']**2 + S_B_filt['xx']*S_B_filt['xz']**2 + S_B_filt['yy']*S_B_filt['xy']**2 + \
                  S_B_filt['zz']*S_B_filt['xz']**2 + S_B_filt['yy']*S_B_filt['yz']**2 + S_B_filt['zz']*S_B_filt['yz']**2) + \
             6 * S_B_filt['xy']*S_B_filt['yz']*S_B_filt['xz']

    return tr_S3

def calc_trace_S_Omegasq_filtered(fname, t, unit, species, win_gauss=0, reselectron=True):

    gradv_filt = compute_gradv_filtered(fname, t, unit, win_gauss, species, reselectron)

    S_filt = pd.DataFrame()
    S_filt.index = gradv_filt.index

    for comp1 in ['x', 'y', 'z']:
        for comp2 in ['x', 'y', 'z']:
            S_filt[f'{comp1}{comp2}'] = 0.5 * (gradv_filt[f'd{comp1}v{comp2}'] + gradv_filt[f'd{comp2}v{comp1}'])

    Omega_filt = compute_Omega_filtered(fname, t, unit, species, win_gauss, reselectron)

    tr_S_Omegasq = -S_filt['xx']*(Omega_filt['xy']**2 + Omega_filt['xz']**2) - 2*S_filt['xy']*Omega_filt['xz']*Omega_filt['yz'] + 2*S_filt['xz']*Omega_filt['xy']*Omega_filt['yz'] \
                   -S_filt['yy']*(Omega_filt['xy']**2 + Omega_filt['yz']**2) - 2*S_filt['yz']*Omega_filt['xy']*Omega_filt['xz'] \
                   -S_filt['zz']*(Omega_filt['xz']**2 + Omega_filt['yz']**2)
    
    return tr_S_Omegasq

def calc_trace_S_Omegasq_B_filtered(fname, t, unit, species, win_gauss=0, reselectron=True):

    gradB_filt = compute_gradB_filtered(fname, t, unit, win_gauss, species, reselectron)

    S_filt = pd.DataFrame()
    S_filt.index = gradB_filt.index

    for comp1 in ['x', 'y', 'z']:
        for comp2 in ['x', 'y', 'z']:
            S_filt[f'{comp1}{comp2}'] = 0.5 * (gradB_filt[f'd{comp1}v{comp2}'] + gradB_filt[f'd{comp2}v{comp1}'])

    Omega_filt = compute_Omega_B_filtered(fname, t, unit, species, win_gauss, reselectron)

    tr_S_Omegasq = -S_filt['xx']*(Omega_filt['xy']**2 + Omega_filt['xz']**2) - 2*S_filt['xy']*Omega_filt['xz']*Omega_filt['yz'] + 2*S_filt['xz']*Omega_filt['xy']*Omega_filt['yz'] \
                   -S_filt['yy']*(Omega_filt['xy']**2 + Omega_filt['yz']**2) - 2*S_filt['yz']*Omega_filt['xy']*Omega_filt['xz'] \
                   -S_filt['zz']*(Omega_filt['xz']**2 + Omega_filt['yz']**2)
    
    return tr_S_Omegasq
