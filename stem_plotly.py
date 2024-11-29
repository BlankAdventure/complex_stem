# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 22:36:47 2024

@author: BlankAdventure
"""

#import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import plotly.io as pio
#from typing import Any, Callable
from utils import tick_formats #, tick_formatter



tf = tick_formats['df_pi']
default_angle = np.deg2rad(80)
sf = 3


def draw_plot(K, fig=None, th=default_angle, sf=3 ):
    N = len(K)
    ki = np.arange(-(N//2),(N+1)//2)
    
    if not fig:
        fig = go.Figure()

    for i in range(N):
        if abs(K[i]) > 1e-3:    
            # real line
            x1 = ki[i]
            x2 = ki[i]
            y1 = 0
            y2 = np.real(K[i])
            fig.add_trace(go.Scatter(x=[x1,x2], y=[y1,y2],marker=dict(color='blue'),
                                     hovertemplate=f"Real: {y2}",name=""))
            
            # imag line
            x1 = ki[i]
            x2 = ki[i]
            y1 = 0
            y2 = np.imag(K[i])
            
            xn = np.cos(th)*(x2-x1) - np.sin(th)*(sf*y2-y1)+x1
            yn = np.sin(th)*(x2-x1) + np.cos(th)*(sf*y2-y1)+y1
            
            
            fig.add_trace(go.Scatter(x=[x1,xn], y=[y1,yn],marker=dict(color='red'),
                                     hovertemplate=f"Imag: {y2}",name=""))
            
            
        fig.add_trace(go.Scatter(x=[ki[i]], y=[0],marker=dict(color='black'),
                                 hovertemplate=tf[0](ki[i],N,0),name=""))
    
    
    fig.update_yaxes(range = [-1,1],showline=True,linecolor='black')
    fig.update_xaxes(range = [ki[0],ki[-1]],showline=True,linecolor='black')
    fig.update_layout(
        xaxis={'anchor': 'free',
               'position': 0.5,
               'tickvals': ki,
               'ticktext': ki
               }, 
        yaxis={'anchor': 'free',
               'position': 0.5,
               'tickvals': [1,-1],
               'ticktext': [1,-1],
               },    
    )
    
    fig.update_layout(showlegend=False, plot_bgcolor='white')
    return fig
   
if __name__ == "__main__":   
    pio.renderers.default='browser'
    N = 21
    K = np.zeros(N,dtype='complex128')
    K[5] = 0.5 + 1j*0.2
    K[15] = -0.8 - 1j*0.4
    
    newfig = draw_plot(K)
    newfig.show()