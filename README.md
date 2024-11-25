# complex_stem
Module for making nice plots of complex DFT vectors

Basic usage:

```python
    from numpy.fft import fft, fftshift
    
    N = 20 #Total number of bins
    k1 = 2.0 #First component
    k2 = 6.0 #Second component
    
    x = np.arange(N)
    s = np.cos(x*2*k1*np.pi/N) + np.sin(x*2*k2*np.pi/N)
    KS = fftshift(fft(s))

    stem2D(KS,fs=150,label_active=True,mode='RI',fancy=True)
```
...which produces the following figure:

<img src="https://github.com/user-attachments/assets/7c8a0547-0c82-461f-a616-8931edf8048f" height="360">

The cosine components show up in bins +/- 2 and fall in the real (vertical) plane. The sine components show up in bins +/- 6, and fall on the imaginary (flat) plane; i.e., looking in/out of the page. The spectrum is hermetian-symmetric, as expected for a purely real input signal.
