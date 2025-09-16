'''
Loads magnetic field data from FGM.
'''

from pyspedas.mms import fgm

def load_fgm(trange, probe="1", drate="brst", wipe=True):

    var_names = [
        'Epoch', 
        'b_gse', 
        'r_gse'
    ]

    data_dict = dict()

    prefix = f"mms{probe}_fgm"
    suffix = f"{drate}_l2"

    # Create list of variables
    
    varlist = {f'{prefix}_{var}_{suffix}': f"{var}_{probe}" for var in var_names}

    # Load FGM data
    data = fgm(
        trange=trange, 
        probe=probe, 
        data_rate=drate,
        varnames=list(varlist.keys()),
        notplot=True
    )

    for key, value in data.items():

        skey = varlist[key]

        data_dict[skey] = dict()

        data_dict[skey]['Epoch'] = data[key]['x']
        data_dict[skey]['Values'] = data[key]['y']
        
    return data_dict
