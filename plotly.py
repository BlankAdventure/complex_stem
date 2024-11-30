# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 22:36:47 2024

@author: BlankAdventure
"""

import plotly.graph_objs as go
import numpy as np
import plotly.io as pio
from utils import tick_formatter, vector

default_angle = np.deg2rad(80)

addbr = lambda x: x.replace('\n','<br>')

class Plotly():
    def __init__(self):
        self.fig = None
        
    def stem(self, K: vector, th:float=default_angle, sf:float=3) -> None:
        self.N = len(K)
        self.ki = np.arange(-(N//2),(N+1)//2)
        self.th = th
        self.sf = sf
        
        if not self.fig:
            self.fig = go.Figure()
            
        self.update(K,show=False)   
        
        self.fig.update_yaxes(range = [-1,1],showline=True,linecolor='black')
        self.fig.update_xaxes(range = [self.ki[0],self.ki[-1]],showline=True,linecolor='black')
        self.fig.update_layout(
            xaxis={'anchor': 'free',
                   'position': 0.5,
                   'tickvals': self.ki,
                   'ticktext': self.ki
                   }, 
            yaxis={'anchor': 'free',
                   'position': 0.5,
                   'tickvals': [1,-1],
                   'ticktext': [1,-1],
                   },    
        )
        self.fig.update_layout(showlegend=False, plot_bgcolor='white')
        self.fig.show()
        
    def update(self, K:vector, show:bool=True) -> None:
        if not self.fig:
            self.stem(K)
        self.fig.data = []
        for i in range(self.N):
            if abs(K[i]) > 1e-3:    
                # real line
                x1 = self.ki[i]
                x2 = self.ki[i]
                y1 = 0
                y2 = np.real(K[i])
                self.fig.add_trace(go.Scatter(x=[x1,x2], y=[y1,y2],marker=dict(color='blue'),
                                         hovertemplate=f"Real: {y2}",name=""))
                # imag line
                x1 = self.ki[i]
                x2 = self.ki[i]
                y1 = 0
                y2 = np.imag(K[i])
                xn = np.cos(self.th)*(x2-x1) - np.sin(self.th)*(self.sf*y2-y1)+x1
                yn = np.sin(self.th)*(x2-x1) + np.cos(self.th)*(self.sf*y2-y1)+y1
                self.fig.add_trace(go.Scatter(x=[x1,xn], y=[y1,yn],marker=dict(color='red'),
                                         hovertemplate=f"Imag: {y2}",name=""))
                
            self.fig.add_trace(go.Scatter(x=[self.ki[i]], y=[0],marker=dict(color='black'),
                                     hovertemplate=addbr(tick_formatter(self.ki[i],self.N,0, methods=['df_pi','df_hz'], units=False)),name=""))

        if show: self.fig.show()
   
if __name__ == "__main__":   
    
    pio.renderers.default='browser'
    
    
    N = 21
    K = np.zeros(N,dtype='complex128')
    K[5] = 0.5 + 1j*0.2
    K[15] = -0.8 - 1j*0.4
    
    newfig = Plotly()
    newfig.stem(K)

    K2 = np.zeros(N,dtype='complex128')
    K2[8] = 0 + 1j*1
    K2[12] = 1 - 1j*0.0
    
    
    newfig.update(K2)
    
