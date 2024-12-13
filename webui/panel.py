# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 00:56:48 2024

@author: Patrick
"""


from collections import defaultdict
from nicegui import ui

# Unpacks val_dict into a nested list-of-lists (each row is a list)
def unpack(val_dict):
    lst = list(val_dict.values())
    table = [lst[i:i + 3] for i in range(0, len(lst), 3)]    
    return table

class Panel():
    def __init__(self, row_def, callback, throttle=0.2):    
        self.val_dict = defaultdict(int)
        self.th = throttle
        self.callback = callback
        self.row_def = row_def        
        self.r_count = 0
        self.e_count = 0
        self.key_list = []

        with ui.row().classes('w-48 p-0 m-0'):
            ui.label('Amp').classes('m-auto p-0 m-0 font-medium')
            ui.label('Freq').classes('m-auto p-0 m-0 font-medium')
            ui.label('Phase').classes('m-auto p-0 m-0 font-medium')     

        c = ui.column().classes('gap-2')
        ui.button('Add', on_click=lambda: self.add_row(c))
        self.add_row(c, do_callback=False)

    def add_element(self, elem):
        self.e_count += 1
        kcoord = f'{self.r_count}-{self.e_count}'        
        elem.bind_value_to(self.val_dict, kcoord)
        elem.on('update:model-value', lambda: self.callback(self.val_dict),throttle=self.th,leading_events=False)

    def delete(self, row):
        for k in list(self.val_dict.keys()):
            if int(k.split('-')[0]) == row:
                del self.val_dict[k]
        self.callback(self.val_dict)

    def add_row(self, col, do_callback=True):
        self.e_count = 0
        self.r_count += 1
        with col:
            with ui.row().classes('items-center') as r:
                [self.add_element(elem()) for elem in self.row_def]
                ui.button(icon='delete', on_click = lambda x=self.r_count: (col.remove(r),self.delete(x))).props('round dense').classes('align-middle')
        if do_callback: self.callback(self.val_dict)
 

