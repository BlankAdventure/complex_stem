# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:52:26 2024

@author: Patrick
"""

import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
from numpy.fft import fft, fftshift
import plotly.graph_objects as go
from nicegui import ui, ElementFilter #, Tailwind
from plotly_stem import PlotlyStem
import numpy as np
#import pandas as pd
from dataclasses import dataclass
from panel import Panel

new_row = {'Amp': 0.5, 'Freq': 2, 'Phase': 0}

@dataclass    
class Config:
    N = 21
    x = np.arange(N)
    
    #df = pd.DataFrame(data={'Amp':[1],'Freq':[1],'Phase':[0],})

def build_sig(groups, x, num_points):
   f = lambda A,F,P: A*np.cos(P*np.pi/180 + 2*np.pi*F*x/num_points)   
   sig = sum([f(*g) for g in groups])
   K = fftshift(fft(sig)) / (num_points//2)
   return (sig, K)

#def build_sig(df, x, num_points):
#    f = lambda A,F,P: A*np.cos(P*np.pi/180 + 2*np.pi*F*x/num_points)
#    sig = sum([f(A, F, P) for A,F,P in zip(df['Amp'], df['Freq'], df['Phase'])])
#    K = fftshift(fft(sig)) / (num_points//2)
#    return (sig, K)

#def update_sig(data):
#    lst = list(data.values())
#    groups = [lst[i:i + 3] for i in range(0, len(lst), 3)]
    
#    f = lambda A,F,P: A*np.cos(P*np.pi/180 + 2*np.pi*F*x/num_points)
    
    
    #data = np.asarray(data)
    #data = np.reshape(a, newshape)
    #print(groups)



def borders_on():
    ElementFilter(kind=ui.column).style('border: solid; border-width: thin; border-color: red;');
    ElementFilter(kind=ui.row).style('border: solid; border-width: thin; border-color: green');


config = Config()


row_def = [lambda: ui.knob(1, show_value=True, track_color='grey-4', 
                           min=0, max=1, step=0.1).props("size=50px :thickness=0.3"),
           lambda: ui.knob(1, show_value=True, track_color='grey-4',
                           min=0, max=config.N, step=1).props("size=50px :thickness=0.3"),
           lambda: ui.knob(0, show_value=True, track_color='grey-4',
                           min=-180, max=180, step=5).props("size=50px :thickness=0.3")]

sig, K = build_sig([[1,1,0]], config.x, config.N)

fig = go.Figure()
zstem = PlotlyStem(fig)
zstem.stem(K)
fig.update_layout(margin=dict(l=20, r=20, t=0, b=10))

#fig2 = go.Figure(go.Scatter(x=config.x, y=sig))

fig2 = go.Figure()
zstem2 = PlotlyStem(fig2)
zstem2.stem(sig)
fig2.update_layout(margin=dict(l=20, r=20, t=0, b=10))

def update(data_in):
    lst = list(data_in.values())
    groups = [lst[i:i + 3] for i in range(0, len(lst), 3)]    
    sig, K = build_sig(groups, config.x, config.N)
    zstem.update(K)
    plot.update()

    zstem2.update(sig)
    time_plot.update()
    #fig2.data[0]['y'] = sig
    #time_plot.update()

with ui.column().classes('w-full'):
    with ui.row().classes('w-full'):
        plot = ui.plotly(fig).classes('w-full h-80')
        #ui.label('-- plotly goes here --')
    with ui.row().classes():
        #ui.label('--- left pane --')
        with ui.column():
            Panel(row_def, update, throttle=0.3)
        
        ui.label('--- right pane --')
        time_plot = ui.plotly(fig2)
        #time_plot = ui

borders_on()

ui.run(port=5000, on_air=False,title='StemPlot',host='0.0.0.0')

