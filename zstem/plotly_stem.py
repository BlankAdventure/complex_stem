# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 22:36:47 2024

@author: BlankAdventure
"""

if __package__:    
  from .utils import tick_formatter, vector
else:
  from utils import tick_formatter, vector

import plotly.graph_objs as go
import numpy as np
import plotly.io as pio

default_angle = np.deg2rad(80)
addbr = lambda x: x.replace('\n','<br>')


RI_opts = {'f1': lambda x: np.real(x),
           'f2': lambda x: np.imag(x),
           'h1': "Real: {y2:.2f}",
           'h2': "Imag: {y2:.2f}"
           } 

MP_opts = {'f1': lambda x: np.abs(x),
           'f2': lambda x: np.angle(x)/np.pi,
           'h1': "Mag: {y2:.2f}",
           'h2': "Ph: {180*y2:.2f}"
           } 

def fstr(template):
    return eval(f"f'{template}'")

class PlotlyStem():
    
    def __init__(self, fig = None):        
        self.__internal = False
        if not fig:
            fig = go.Figure()
            self.__internal = True
        self.fig = fig
        

    def set_mode(self, mode: str) -> None:
        match mode:
            case "MP":
                self.opts = MP_opts
            case "RI":
                self.opts = RI_opts        

        
    def stem(self, K: vector, th:float=default_angle, sf:float=2.5, mode:str='MP') -> go.Figure:
        self.N = len(K)
        self.ki = np.arange(-(self.N//2),(self.N+1)//2)
        self.th = th
        self.sf = sf
        self.mode = mode
            
        self.__redraw(K)
        
        self.fig.update_yaxes(range = [-1,1],showline=True,linecolor='black')
        self.fig.update_xaxes(range = [self.ki[0],self.ki[-1]],showline=True,linecolor='black')
        self.fig.update_layout(
            xaxis={'anchor': 'free',
                   'position': 0.5,
                   'tickvals': self.ki,
                   'ticktext': self.ki,
                   'range': [self.ki[0]-0.10,self.ki[-1]+0.10]
                   }, 
            yaxis={'anchor': 'free',
                   'position': 0.5,
                   'tickvals': [1,-1],
                   'ticktext': [1,-1],
                   },    
        )
        self.fig.update_layout(showlegend=False, plot_bgcolor='white')
        if self.__internal: self.fig.show()
        return self.fig
        
    def __redraw(self, K:vector) -> None:
        self.fig.data = []
        for i in range(self.N):
            if abs(K[i]) > 1e-3:                    
                x1 = self.ki[i]
                x2 = self.ki[i]
                y1 = 0
                
                # ***** Real or Magnitude stem *****                
                y2 = self.opts['f1'](K[i])
                self.fig.add_trace(go.Scatter(x=[x1,x2], y=[y1,y2],marker=dict(color='blue'),
                                         hovertemplate=eval(f"f'{self.opts['h1']}'"),name=""))
                
                
                # ***** Imag or Phase stem *****                
                y2 = self.opts['f2'](K[i])
                xn = np.cos(self.th)*(x2-x1) - np.sin(self.th)*(self.sf*y2-y1)+x1
                yn = np.sin(self.th)*(x2-x1) + np.cos(self.th)*(self.sf*y2-y1)+y1
                self.fig.add_trace(go.Scatter(x=[x1,xn], y=[y1,yn],marker=dict(color='red'),
                                         hovertemplate=eval(f"f'{self.opts['h2']}'"),name=""))
                
            self.fig.add_trace(go.Scatter(x=[self.ki[i]], y=[0],marker=dict(color='black'),
                                     hovertemplate=addbr(tick_formatter(self.ki[i],self.N,0, methods=['df_pi','df_hz'], units=False)),name=""))

    def update(self, K:vector) -> go.Figure:
        self.__redraw(K)
        if self.__internal: self.fig.show()
        return self.fig
    
    mode = property(fset=set_mode)        
   
if __name__ == "__main__":   
    
    pio.renderers.default='browser'
    
    
    N = 21
    K = np.zeros(N,dtype='complex128')
    K[5] = 0.5 + 1j*0.2
    K[15] = -0.8 - 1j*0.4
    
    newfig = PlotlyStem()
    newfig.stem(K)

    K2 = np.zeros(N,dtype='complex128')
    K2[8] = 0 + 1j*1
    K2[12] = 1 - 1j*0.0
    
    
    newfig.update(K2)
    
