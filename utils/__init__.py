from . import data_constants

def get_data_constants(name):
    name = name.upper()

    if name == "EURO": 
        wlc_mean = getattr(data_constants, f'{name}_WLC_MEAN')
        wlc_std = getattr(data_constants, f'{name}_WLC_STD')
        blc_mean = getattr(data_constants, f'{name}_BLC_MEAN')
        blc_std = getattr(data_constants, f'{name}_BLC_STD')
    else:
        raise ValueError(f"Dataset {name} not found.")
        
    return wlc_mean, wlc_std, blc_mean, blc_std

