import pandas as pd
import numpy as np

def dot(a, b): # Create dot product of two vector time series

    adotb = (a.values * b.values).sum(axis=1)

    df = pd.DataFrame(adotb, index=a.index)

    return df

def cross(a, b):

    acrossb = np.cross(a.values , b.values)

    df = pd.DataFrame(acrossb, index=a.index, columns=['x', 'y', 'z'])

    return df

def mult(a, b): #Create dyadic product of two vector time series

    coords = 'xyz'

    a_cols = a.columns  # assumed to be ordered like x, y, z
    b_cols = b.columns  # assumed to be ordered like x, y, z

    ab = pd.DataFrame(columns=['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz'])

    for comp1 in ['x', 'y', 'z']:
        for comp2 in ['x', 'y', 'z']:

            ab[f'{comp1}{comp2}'] = a[a_cols[coords.find(comp1)]] * b[b_cols[coords.find(comp2)]]

    return ab

def grad(k, v):
    
    gradv = mult(k[1], v[1])
    
    for probe in [2, 3, 4]:
        
        gradv += mult(k[probe], v[probe])
        
    return gradv

def curl(k, v):
    
    curlv = cross(k[1], v[1])
    
    for probe in [2, 3, 4]:
        
        curlv += cross(k[probe], v[probe])
        
    return curlv

def div(k, v):
    
    divv = dot(k[1], v[1])
    
    for probe in [2, 3, 4]:
        
        divv += dot(k[probe], v[probe])

def dotgrad(k, A): # Computes A dot grad, A is a tensor of rank 2

    Adotgrad = pd.DataFrame(np.zeros(k[1].shape), columns=['x', 'y', 'z'])

    Adotgrad.index = k[1].index

    for probe in [1, 2, 3, 4]:
        
        for comp1 in ['x', 'y', 'z']:

            for comp2 in ['x', 'y', 'z']:

                Adotgrad[f'{comp1}'] += A[probe][f'{comp1}{comp2}'] * k[probe][f'{comp2}']

    return Adotgrad

