'''
Loads magnetic field data from FGM.
'''

from pyspedas.mms import edp

def load_edp(trange, probe="1", drate="brst", wipe=True):

    var_names = [
        'Epoch', 
        'edp_dce_gse'
    ]

    data_dict = dict()

    prefix = f"mms{probe}"
    suffix = f"{drate}_l2"

    # Create list of variables
    
    varlist = {f'{prefix}_{var}_{suffix}': f"{var}_{probe}" for var in var_names}

    # Load FGM data
    data = edp(
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
