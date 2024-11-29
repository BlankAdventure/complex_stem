import numpy as np
from nptyping import NDArray, Shape
from typing import Any, Callable
from dataclasses import dataclass, field

vector =  NDArray[Shape["*"], Any]


tick_formats: dict[str, tuple[Callable, str]] = {
    'f_hz':   (lambda *a: f'\n{a[2]*a[0]/a[1]:.0f}', '[Hz]'),
    'df_hz':  (lambda *a: f'\n{a[0]/a[1]:.2f}', '[cyc/samp]'),
    'df_rad': (lambda *a: f'\n{2*a[0]/a[1]:.2f}π)', '[rad/samp]'),
    'df_pi':  (lambda *a: f'\n{2*a[0]}π/{a[1]}', '[rad/samp]')
}
             



@dataclass    
class DefaultConfigDict:
    fs: float|None = None
    angle: float = 85
    sf: float = 2
    tick_format: list = field(default_factory=lambda: ['df_pi','df_hz'])
    mag_limit: float = 1e-3
    figsize: tuple[float,float] =(6,4)
    norm: str|None='bin'
    label_active: bool = False
    mode: str|None = 'bin'
    fancy: bool = True
    real_mag: str = 'blue'
    imag_ph: str = 'red'

def merge_dicts(base_dict: dict, kw_dict: dict) -> dict:
    '''
    

    Parameters
    ----------
    base_dict : dict
        Base/reference dict. This defines the valid set of keys.
    kw_dict : dict
        Dictionary with new key values.

    Raises
    ------
    KeyError
        Keys in kw_dict not present in base_dict will raise this error.

    Returns
    -------
    dict
        A new dictionary where common keys will take their values from
        kw_dict. Unmatched keys in base_dict retain their values.

    '''
    # Create a copy of base_dict to avoid mutating the original
    merged_dict = base_dict.copy()
    # Check for unmatched keys in in_dict
    for key in kw_dict:
        if key not in base_dict:
            raise KeyError(f"Key '{key}' from kw_dict is not present in base_dict.")
        # If key exists in both, overwrite with in_dict's value
        merged_dict[key] = kw_dict[key]
    return merged_dict


def get_index(N:int) -> vector:
    '''  
    Parameters
    ----------
    N : int
        Number of frequency bins (two-sided sense)

    Returns
    -------
    vector
        DC-centered frequency bin indicies.

    '''
    return np.arange(-(N//2),(N+1)//2)



def tick_formatter(k:int,N:int,fs:None|float=None, methods: list=[str], units:bool=False) -> str:
    '''
    Parameters
    ----------
    k : int
        k-th bin number 
    N : int
        Total number of bins
    fs : None|float, optional
        Sampling rate in [Hz], if desired.
    methods : list[Callable], optional
        List of tick formatters. Each tick formatter will
        produced an additional text output in the lable. The default is [].        

    Returns
    -------
    str
        The formatted tick label string.

    '''   
    
    out_str = ''.join([tick_formats[key][0](k,N,fs)+tick_formats[key][1] if units else tick_formats[key][0](k,N,fs) for key in methods])
    return out_str.strip()
