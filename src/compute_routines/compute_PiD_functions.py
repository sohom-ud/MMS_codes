import pandas as pd
import numpy as np
from src.utils.hdf_to_df import hdf_to_df
from src.utils.resample import resample

# Computes the Pi tensor from .h5 file

def compute_gradv(fname, species='ion', reselectron=True):

    # if species=='ion' and reselectron:
    #     varlist = [f'v_spincorr_{species}_{probe}_reselectron' for probe in [1, 2, 3, 4]] + [f'k_{probe}_res_ve' for probe in [1, 2, 3, 4]]
    # else:
    #     varlist = [f'v_spincorr_{species}_{probe}' for probe in [1, 2, 3, 4]] + [f'k_{probe}_res_vi' for probe in [1, 2, 3, 4]]

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

    gradv_df = pd.DataFrame()
    gradv_df.index = k[1].index

    for comp1 in ['x', 'y', 'z']:

        for comp2 in ['x', 'y', 'z']:

            gradv_df[f'd{comp1}v{comp2}'] = 0.0

            for probe in [1, 2, 3, 4]:

                gradv_df[f'd{comp1}v{comp2}'] += k[probe][comp1] * v[probe][comp2]

    return gradv_df

def compute_avg_P(fname, species='ion', reselectron=True):

    if species=='ion' and reselectron:
        varlist = [f'Ptensor_{species}_{probe}_reselectron' for probe in [1, 2, 3, 4]]
    else:
        varlist = [f'Ptensor_{species}_{probe}' for probe in [1, 2, 3, 4]]

    df_dict = hdf_to_df(fname, vars=varlist)

    P = dict()
    Pavg_df = pd.DataFrame()

    if species=='ion' and reselectron:
        Pavg_df.index = df_dict[f'Ptensor_{species}_1_reselectron'].index
        for probe in [1, 2, 3, 4]:
            P[probe] = df_dict[f'Ptensor_{species}_{probe}_reselectron']
    else:
        Pavg_df.index = df_dict[f'Ptensor_{species}_1'].index
        for probe in [1, 2, 3, 4]:
            P[probe] = df_dict[f'Ptensor_{species}_{probe}']

    Pavg_df = (P[1] + P[2] + P[3] + P[4])/4.0

    return Pavg_df

def compute_p(fname, species='ion', reselectron=True):

    Pavg = compute_avg_P(fname, species, reselectron)

    p = (Pavg['xx'] + Pavg['yy'] + Pavg['zz'])/3.0

    return p

def compute_theta(fname, species='ion', reselectron=True):

    gradv_df = compute_gradv(fname, species, reselectron)

    theta = (gradv_df['dxvx'] + gradv_df['dyvy'] + gradv_df['dzvz'])

    return theta

def compute_ptheta(fname, species='ion', reselectron=True):

    p = compute_p(fname, species, reselectron)
    theta = compute_theta(fname, species, reselectron)

    ptheta = pd.DataFrame()

    ptheta.index = p.index

    ptheta[f'ptheta_{species}'] = p.values * theta.values

    return ptheta

def compute_Dij(fname, species='ion', reselectron=True):

    gradv_df = compute_gradv(fname, species, reselectron)
    theta = compute_theta(fname, species, reselectron)

    D = pd.DataFrame()

    D.index = gradv_df.index

    for comp1 in ['x', 'y', 'z']:

        for comp2 in ['x', 'y', 'z']:

            D[f'{comp1}{comp2}'] = (gradv_df[f'd{comp1}v{comp2}'] + gradv_df[f'd{comp2}v{comp1}'])/2.0 - (theta * (comp1 == comp2))/3.0

    return D

def compute_Pi_ij(fname, species='ion', reselectron=True):

    Pavg = compute_avg_P(fname, species, reselectron)
    p = compute_p(fname, species, reselectron)

    Pi_ij = pd.DataFrame()
    Pi_ij.index = Pavg.index

    for comp1 in ['x', 'y', 'z']:
        
        for comp2 in ['x', 'y', 'z']:

            Pi_ij[f'{comp1}{comp2}'] = Pavg[f'{comp1}{comp2}'] - (p * (comp1 == comp2))

    return Pi_ij

def compute_PiD(fname, species='ion', reselectron=True):
    '''
    Returns value of Pi-D without the negative sign included.
    '''
    Pi = compute_Pi_ij(fname, species, reselectron)
    D = compute_Dij(fname, species, reselectron)

    PiD = pd.DataFrame()

    PiD.index = Pi.index
    PiD[f'PiD_{species}'] = 0.0

    for comp1 in ['x', 'y', 'z']:

        for comp2 in ['x', 'y', 'z']:

            PiD[f'PiD_{species}'] += Pi[f'{comp1}{comp2}'] * D[f'{comp1}{comp2}']

    return PiD

def compute_PS(fname, species='ion', reselectron=True):
    '''
    Returns value of PS without negative sign included.
    '''
    PiD = compute_PiD(fname, species, reselectron)
    ptheta = compute_ptheta(fname, species, reselectron)

    PS = pd.DataFrame()

    PS.index = PiD.index

    PS[f'PS_{species}'] = PiD.values + ptheta.values

    return PS

def compute_jcurl(fname):

    mu0 = 4 * np.pi * 1e-7

    df_dict = hdf_to_df(fname)

    k = dict()
    b = dict()

    j = 0.0

    for probe in [1, 2, 3, 4]:

        k[probe] = df_dict[f'k_{probe}_res_ve']
        b[probe] = resample(df_dict[f'b_gse_{probe}'].iloc[:, :3], df_dict['v_spincorr_elc_1'])

        j += np.cross(k[probe], b[probe])

    j = (j * 1e-12 / mu0)/1e-9
    
    j_df = pd.DataFrame(data=j, index=b[1].index, columns=['x', 'y', 'z'])

    return j_df

def compute_jdotE(fname):

    j = compute_jcurl(fname)

    df_dict = hdf_to_df(fname)

    E = dict()

    jdotE = 0.0

    for probe in [1, 2, 3, 4]:

        E[probe] = df_dict[f'edp_dce_gse_{probe}']

    jdotE = ((j * E[1]).sum(axis=1) + (j * E[2]).sum(axis=1) + (j * E[3]).sum(axis=1) + (j * E[4]).sum(axis=1))/4000.0

    jdotE_df = pd.DataFrame(data=jdotE, index=j.index, columns=['jdotE'])

    return jdotE_df

def compute_jpart(fname):

    e = 1.6e-19

    df_dict = hdf_to_df(fname)
    j = pd.DataFrame()

    ni = df_dict['N_ion_1']
    vi = df_dict['v_spincorr_ion_1']
    ve = df_dict['v_spincorr_elc_1']

    ve = resample(ve, vi)

    j = e * (vi-ve).mul(ni.values, axis=0)

    return j

def compute_Q_D(fname, species='ion', reselectron=True):

    Dij = compute_Dij(fname, species, reselectron)

    Dsq = 0.0

    for comp1 in ['x', 'y', 'z']:
        for comp2 in ['x', 'y', 'z']:
            Dsq += Dij[f'{comp1}{comp2}'] ** 2
    
    Dsqmean = np.mean(Dsq)

    Q_D = (1/4.) * Dsq / Dsqmean

    return Q_D

def compute_Q_omega(fname, species='ion', reselectron=True):

    df_dict = hdf_to_df(fname)

    k = dict()
    v = dict()

    omega = 0.0

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

        omega += np.cross(k[probe], v[probe])

    omega_df = pd.DataFrame(data=omega, index=v[1].index, columns=['x', 'y', 'z'])

    omegasq = omega_df['x']**2 + omega_df['y']**2 + omega_df['z']**2
    omegasqmean = np.mean(omegasq)

    Q_omega = (1/4.) * omegasq / omegasqmean

    return Q_omega

def compute_Q_j(fname):

    j = compute_jcurl(fname)
    # j = compute_jpart(fname)

    jsq = j['x']**2 + j['y']**2 + j['z']**2
    jsqmean = np.mean(jsq)

    Q_j = (1/4.) * jsq / jsqmean

    return Q_j

# def compute_R_D(fname, species='ion', reselectron=True):

#     Dij = compute_Dij(fname, species, reselectron)

#     R = (Dij['xx'] * (Dij['yy'] * Dij['zz'] - Dij['yz'] ** 2)
#         - Dij['xy'] * (Dij['yx'] * Dij['zz'] - Dij['zx'] * Dij['yz'])
#         + Dij['xz'] * (Dij['yx'] * Dij['zy'] - Dij['zx'] * Dij['yy']))
    
#     R / = np.std(R)

#     return R