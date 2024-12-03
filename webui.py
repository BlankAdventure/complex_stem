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
from dataclasses import dataclass
from panel import Panel, unpack
from functools import partial


class DFTPlot():
    def __init__(self, K):
        self.dft_fig = go.Figure()
        self.zstem = PlotlyStem(self.dft_fig)
        self.zstem.stem(K)
        self.dft_fig.update_layout(margin=dict(l=20, r=20, t=0, b=10))
        self.dft_plot = ui.plotly(self.dft_fig).classes('w-full h-80')

    def update(self, new_K):
        self.zstem.update(new_K)
        self.dft_plot.update()


class Signal():
    def __init__(self, N):
        self.N = N
        self.x = np.arange(N)
        self.y = None
        self.K = None
    def build_sig(self, src_list):
        sig = sum([g[0]*np.cos(g[2]*np.pi/180 + 2*np.pi*g[1]*self.x/self.N) for g in src_list])
        K = fftshift(fft(sig)) / (self.N//2)
        self.y = sig
        self.K = K
        return (sig, K)


def borders_on():
    ElementFilter(kind=ui.column).style('border: solid; border-width: thin; border-color: red;');
    ElementFilter(kind=ui.row).style('border: solid; border-width: thin; border-color: green');



N = 21

row_def = [lambda: ui.knob(1, show_value=True, track_color='grey-4', 
                           min=0, max=1, step=0.1).props("size=50px :thickness=0.3"),
           lambda: ui.knob(1, show_value=True, track_color='grey-4',
                           min=0, max=N, step=1).props("size=50px :thickness=0.3"),
           lambda: ui.knob(0, show_value=True, track_color='grey-4',
                           min=-180, max=180, step=5).props("size=50px :thickness=0.3")]
#sig = Signal(N)

#y, K = sig.build_sig([[1,1,0]])

#dft_plot = DFTPlot(K)

#sig, K = build_sig([[1,1,0]], config.x, config.N)
#callback = partial(update, dft_obj = dft_plot)
#fig2 = go.Figure(go.Scatter(x=config.x, y=sig))

#fig2 = go.Figure()
#zstem2 = PlotlyStem(fig2)
#zstem2.stem(sig)
#fig2.update_layout(margin=dict(l=20, r=20, t=0, b=10))

class App():
    def __init__(self):
        self.sig = Signal(N)
        self.sig.build_sig([[1,1,0]])
        self.setup_ui()
    
    def update(self, new_data):
        sources = unpack(new_data)
        y,K = self.sig.build_sig(sources)
        self.dft_plot.update(K)

    def setup_ui(self):
        with ui.column().classes('w-full'):
            with ui.row().classes('w-full'):
                # ***** dft plot here ****
                self.dft_plot = DFTPlot(self.sig.K)
        
            with ui.row().classes():
                #ui.label('--- left pane --')
                with ui.column():
                    Panel(row_def, callback=self.update, throttle=0.3)
                
                ui.label('--- right pane --')
                #time_plot = ui.plotly(fig2)
                #time_plot = ui
App()
borders_on()

ui.run(port=5000, on_air=False,title='StemPlot',host='0.0.0.0')

