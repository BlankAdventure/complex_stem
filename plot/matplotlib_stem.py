"""
Creates a nice visualization of an FFT signal represenation.

"""
if __package__:    
  from .utils import DefaultConfigDict, vector, merge_dicts, tick_formatter, get_index
else:
  from utils import DefaultConfigDict, vector, merge_dicts, tick_formatter, get_index



import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from dataclasses import asdict

TICK_BACKGROUND = "#FFFFFFBF"

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
