"""
Creates a nice visualization of an FFT signal represenation.

"""


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from nptyping import NDArray, Shape
from typing import Any, Callable

vector =  NDArray[Shape["*"], Any]


tick_formats: dict[str, tuple[Callable, str]] = {
    'f_hz':   (lambda *a: f'\n{a[2]*a[0]/a[1]:.0f}', '[Hz]'),
    'df_hz':  (lambda *a: f'\n{a[0]/a[1]:.2f}', '[cyc/samp]'),
    'df_rad': (lambda *a: f'\n{2*a[0]/a[1]:.2f}π)', '[rad/samp]'),
    'df_pi':  (lambda *a: f'\n{2*a[0]}π/{a[1]}', '[rad/samp]')
}
             
default_ticks = ['df_pi', 'df_hz']

MAG_LIMIT = 1e-3
TICK_BACKGROUND = "#FFFFFFBF"

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

def stem2D(K:vector, fs:None|float=None, angle:float=80, sf:float=1.5, \
                label_active:bool=False, mode:str='RI', normalize:bool=True, fancy:bool=True, \
                format_list: list = default_ticks, mag_limit:float=MAG_LIMIT, \
                figsize: tuple[float,float] =(6,4)) -> None:
    '''
    Parameters
    ----------
    K : vector
        Vector of frequency components. Should be DC-centered!
    fs : None|float, optional
        Sampling rate, to display on x-axis, if desired.
        The default is None.
    angle : float, optional
        Angle at which to draw the imag or phase components. 
        The default is 80.
    sf : float, optional
        Optional "stretch factor" applied to imag or phase components. 
        The default is 1.5.
    label_active : bool, optional
        Label the ticks corresponding to active components. 
        The default is False.
    mode : str, optional
        Components can be real/imaginary (RI) or mag/phase (MP). 
        The default is 'RI'.

    Returns
    -------
    None
        Displays the plot.

    '''

    th = np.deg2rad(angle)
    Nk = len(K)
    xx = np.arange(Nk)    
    ki = get_index(Nk)
    format_list = format_list[:]
    
    if normalize:
        K = K / max(abs(K))
        
    if fs:
        format_list.append('f_hz')  
    
    if mode == 'MP':
        rr = np.abs(K)
        ii = np.angle(K)        
        legend_names = ['Mag', 'Phase']
    else:
        rr = np.real(K)
        ii = np.imag(K)
        legend_names = ['Real', 'Imag']
        

    tick_pos = []
    tick_labels = []

    if fancy: #apppend left-side tick labels
        tick_pos.append(0)
        tick_labels.append( tick_formatter(ki[0],Nk,fs=fs,methods=format_list,units=True) )
    
    fig, ax = plt.subplots(layout='constrained', figsize=figsize)

    for x in xx:        
        if abs(K[x]) > mag_limit:            
            # ****** imaginary or phase lines *****
            x1 = x
            y1 = 0
            
            x2 = x
            y2 = sf*ii[x]
            # We need to rotate the point to produce the illusion of lying
            # in the flat plane
            xn = np.cos(th)*(x2-x1) - np.sin(th)*(y2-y1)+x1
            yn = np.sin(th)*(x2-x1) + np.cos(th)*(y2-y1)+y1
            
            xi = (x1, xn)
            yi = (y1, yn)
            
            ax.plot(xi, yi, color='red', linestyle='dotted', zorder=1)
            ax.scatter(xi[1], yi[1], marker='.', color='red', zorder=1)
        
            # ****** real or magnitude lines *****
            xr = (x, x)
            yr = (0, rr[x])
            
            ax.plot(xr, yr, color='blue', zorder=2)
            ax.scatter(xr[1], yr[1], marker='.', color='blue', zorder=2)    
    
            # get active tick lables, if enabled
            if label_active:
                tick_pos.append(x)
                tick_labels.append(tick_formatter(ki[x],Nk,fs=fs,methods=format_list))
        elif fancy: ax.scatter(x,0, marker='.', color='black', zorder=0)
    
    if normalize:
        ax.set_ylim([-1.02,1.02])        
        plt.yticks(ticks=[-1,1],labels=["-1","1"])    
    
    if fancy:
        tick_pos.append(Nk-1)
        tick_labels.append(tick_formatter(ki[-1],Nk,fs=fs,methods=format_list,units=True))
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
    colors = ['blue','red']
    lines = [Line2D([0], [0], color=c, linewidth=2, linestyle='-') for c in colors]    
    plt.legend(lines, legend_names, fontsize = 'x-small', loc=2)
    

if __name__ == "__main__":
    # Demo run
    
    from numpy.fft import fft, fftshift
    
    N = 20 #Total number of bins
    k1 = 2.0 #First component
    k2 = 6.0 #Second component
    
    x = np.arange(N)
    s = np.cos(x*2*k1*np.pi/N) + np.sin(x*2*k2*np.pi/N)
    KS = fftshift(fft(s))

    stem2D(KS,fs=150,label_active=False,mode='MP',fancy=True, figsize=(5,3))
