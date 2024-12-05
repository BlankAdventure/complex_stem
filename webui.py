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
from plotly_stem import PlotlyStem, vector
import numpy as np
from panel import Panel, unpack



class AliasKnob(ui.knob):
    def __init__(self, *args, limit:int=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit = limit
        self.on_value_change(self.check_alias)
        
    def check_alias(self, e):
        if self.value > self.limit:
            self.props ("color=red")
        else:
            self.props ("color=blue")

class TimePlot():
    def __init__(self,x:vector, y:vector):
        self.time_fig = go.Figure()        
        self.time_fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers',
                                           line = dict(color='lightgrey',dash='dash'),
                                           marker = dict(color='blue'),name=""))
        self.time_fig.add_trace(go.Scatter(x=x, y=np.zeros(len(x)),mode='markers',marker=dict(color='black'),name=""))
                              
        self.time_fig.update_yaxes(range = [-1,1],showline=True,linecolor='black')
        self.time_fig.update_xaxes(range = [x[0],x[-1]],showline=True,linecolor='black')
        self.time_fig.update_layout(
            xaxis={'anchor': 'free',
                   'position': 0.5,
                   'tickvals': x,
                   'ticktext': x,
                   'range': [x[0]-0.10,x[-1]+0.10]
                   }, 
            yaxis={'anchor': 'free',
                   'position': 0.0,
                   'tickvals': [1,-1],
                   'ticktext': [1,-1],
                   },    
        )
        self.time_fig.update_layout(showlegend=False, plot_bgcolor='white')        
        self.time_fig.update_layout(margin=dict(l=20, r=20, t=0, b=10))
        self.time_plot = ui.plotly(self.time_fig).classes('w-full h-80')
    
        
    def update(self, y_new: vector):
        self.time_fig['data'][0]['y'] = y_new
        self.time_plot.update()
        
class DFTPlot():
    def __init__(self, K:vector):
        self.dft_fig = go.Figure()
        self.zstem = PlotlyStem(self.dft_fig)
        self.zstem.stem(K)
        self.dft_fig.update_layout(margin=dict(l=20, r=20, t=0, b=10))
        self.dft_plot = ui.plotly(self.dft_fig).classes('w-full h-80')

    def update(self, new_K:vector):
        self.zstem.update(new_K)
        self.dft_plot.update()

def build_sig(src_list: list[list], N:int, x: vector):
    sig = sum([g[0]*np.cos(g[2]*np.pi/180 + 2*np.pi*g[1]*x/N) for g in src_list])
    K = fftshift(fft(sig)) / (N//2)
    return (sig, K)


def borders_on():
    ElementFilter(kind=ui.column).style('border: solid; border-width: thin; border-color: red;');
    ElementFilter(kind=ui.row).style('border: solid; border-width: thin; border-color: green');



N = 21

row_def = [lambda: ui.knob(0.5, show_value=True, track_color='grey-4', 
                           min=0, max=1, step=0.1).props("size=50px :thickness=0.3"),
           lambda: AliasKnob(1, limit=N//2, show_value=True, track_color='grey-4',
                           min=0, max=N, step=1).props("size=50px :thickness=0.3"),
           lambda: ui.knob(0, show_value=True, track_color='grey-4',
                           min=-180, max=180, step=5).props("size=50px :thickness=0.3")]

class App():
    def __init__(self, N:int=N):
        self.N = N
        self.x = np.arange(N)
        self.setup_ui()
    
    def update(self, new_data: dict):        
        sources = unpack(new_data)        
        sig,K = build_sig(sources,self.N,self.x)
        self.dft_plot.update(K)
        self.time_plot.update(sig)
        
    def setup_ui(self):
        sig, K = build_sig([[0.5,1,0]], self.N, self.x)        
        with ui.row().classes('mx-auto'):
            with ui.column():
                with ui.card().classes('h-64'):
                    self.dft_plot = DFTPlot(K)                    
                with ui.card().classes('h-64'):
                    self.time_plot = TimePlot(self.x, sig)
            with ui.column():
                with ui.card():
                    Panel(row_def, callback=self.update, throttle=0.15)
        
App()
#borders_on()

ui.run(port=5000,title='DFT Plot',host='0.0.0.0')

