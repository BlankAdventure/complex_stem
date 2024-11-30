# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:52:26 2024

@author: Patrick
"""

import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import plotly.graph_objects as go
from nicegui import ui #, Tailwind
from plotly_stem import PlotlyStem
import numpy as np
#from functools import wraps, partial



N = 21
K = np.zeros(N,dtype='complex128')
K[5] = 0.5 + 1j*0.2
K[15] = -0.8 - 1j*0.4

fig = go.Figure()
zstem = PlotlyStem(fig)
zstem.stem(K)
fig.update_layout(margin=dict(l=20, r=20, t=0, b=10))


with ui.column().classes('w-full').style('border: solid; border-width: thin;'):
    with ui.row().classes('w-full').style('border: solid; border-width: thin;'):
        plot = ui.plotly(fig).classes('w-full h-80')
        #ui.label('-- plotly goes here --')
    with ui.row().classes().style('border: solid; border-width: thin;'):
        ui.label('--- left pane --')
        ui.label('--- right pane --')




ui.run(port=5000, on_air=False,title='CircFit',host='0.0.0.0')

