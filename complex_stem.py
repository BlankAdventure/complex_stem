"""
Creates a nice visualization of an FFT signal represenation.

"""


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from nptyping import NDArray, Shape
from typing import Any, Callable
from dataclasses import dataclass, field, asdict

vector =  NDArray[Shape["*"], Any]


tick_formats: dict[str, tuple[Callable, str]] = {
    'f_hz':   (lambda *a: f'\n{a[2]*a[0]/a[1]:.0f}', '[Hz]'),
    'df_hz':  (lambda *a: f'\n{a[0]/a[1]:.2f}', '[cyc/samp]'),
    'df_rad': (lambda *a: f'\n{2*a[0]/a[1]:.2f}π)', '[rad/samp]'),
    'df_pi':  (lambda *a: f'\n{2*a[0]}π/{a[1]}', '[rad/samp]')
}
             

TICK_BACKGROUND = "#FFFFFFBF"

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

def stem2D(K:vector, config=DefaultConfigDict(), **kwargs) -> None:
    '''
    

    Parameters
    ----------
    K : vector
        The input DFT vector (DC-centered!)
    config : TYPE, optional
        Plotting configuration options. 
        The default is DefaultConfigDict().        
    **kwargs : TYPE
        Any plotting arguments from DefaultConfigDict()

    Returns
    -------
    None
        Generates a nice-looking DFT plot.

    '''
    
    
    
    config = merge_dicts(asdict(config), kwargs)

    th = np.deg2rad(config['angle'])
    Nk = len(K)
    xx = np.arange(Nk)    
    ki = get_index(Nk)
    format_list = config['tick_format']

    match config['norm']:
        case 'mag':
            K = K / max(abs(K))
        case 'bin':
            K = K / (Nk//2)        
        
    if config['fs']:
        format_list.append('f_hz')  
    
    if config['mode'] == 'MP':
        rr = np.abs(K)
        ii = np.angle(K)
        legend_names = ['Mag', 'Phase']
    else:
        rr = np.real(K)
        ii = np.imag(K)
        legend_names = ['Real', 'Imag']
        

    tick_pos = []
    tick_labels = []

    if config['fancy']: #apppend left-side tick labels
        tick_pos.append(0)
        tick_labels.append( tick_formatter(ki[0],Nk,fs=config['fs'],methods=format_list,units=True) )
    
    _, ax = plt.subplots(layout='constrained', figsize=config['figsize'])

    for x in xx:        
        if abs(K[x]) > config['mag_limit']:            
            # ****** imaginary or phase lines *****
            x1 = x
            y1 = 0
            
            x2 = x
            y2 = config['sf']*ii[x]
            # We need to rotate the point to produce the illusion of lying
            # in the flat plane
            xn = np.cos(th)*(x2-x1) - np.sin(th)*(y2-y1)+x1
            yn = np.sin(th)*(x2-x1) + np.cos(th)*(y2-y1)+y1
            
            xi = (x1, xn)
            yi = (y1, yn)
            
            ax.plot(xi, yi, color=config['imag_ph'], linestyle='dotted', zorder=1)
            ax.scatter(xi[1], yi[1], marker='.', color=config['imag_ph'], zorder=1)
        
            # ****** real or magnitude lines *****
            xr = (x, x)
            yr = (0, rr[x])
            
            ax.plot(xr, yr, color=config['real_mag'], zorder=2)
            ax.scatter(xr[1], yr[1], marker='.', color=config['real_mag'], zorder=2)    
    
            # get active tick lables, if enabled
            if config['label_active']:
                tick_pos.append(x)
                tick_labels.append(tick_formatter(ki[x],Nk,fs=config['fs'],methods=format_list))
        elif config['fancy']: ax.scatter(x,0, marker='.', color='black', zorder=0)
    
    # Scale y-axis based on max component magnitude        
    y_ext = np.max(np.abs(K))
    ax.set_ylim([-y_ext*1.02,y_ext*1.02])        
    plt.yticks(ticks=[-y_ext,y_ext],labels=[f'{-y_ext:.1f}',f'{y_ext:.1f}'])    
    
    if config['fancy']:
        tick_pos.append(Nk-1)
        tick_labels.append(tick_formatter(ki[-1],Nk,fs=config['fs'],methods=format_list,units=True))
        ax.set_xticks(xx, ki)   
        ax.tick_params(labelsize=8)
    else:
        ax.set_xticks([])

    ax.set_xlim([-0.1, Nk-1+0.1])    
    sec = ax.secondary_xaxis(location=0.50, zorder=3)    
    sec.set_xticks(tick_pos, tick_labels, zorder=3 )    
    sec.tick_params(labelsize=8)
    plt.setp( ax.get_xticklabels(), backgroundcolor=TICK_BACKGROUND)
    plt.setp(sec.get_xticklabels(), backgroundcolor=TICK_BACKGROUND)   

        
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position(('data',Nk//2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Make a custom legend
    colors = [config['real_mag'],config['imag_ph']]
    lines = [Line2D([0], [0], color=c, linewidth=2, linestyle='-') for c in colors]    
    plt.legend(lines, legend_names, fontsize = 'x-small', loc=2)
    
    plt.show()
    

if __name__ == "__main__":
    # Demo run
    
    from numpy.fft import fft, fftshift
    
    N = 20 #Total number of bins
    k1 = 2.0 #First component
    k2 = 6.0 #Second component
    
    x = np.arange(N)
    s = np.cos(x*2*k1*np.pi/N) + np.sin(x*2*k2*np.pi/N)
    KS = fftshift(fft(s))

    stem2D(KS,fs=150,label_active=False,mode='MP',fancy=True, figsize=(5,3) )
