"""
This file was created in order to bring
common variables and functions into one file to make
code more clear
"""
from pathlib import Path


ascii_offset = 161
beam_size = 3
th_reduction_factor = 1

root_folder = Path.cwd() / 'implementation_synthetic_logs'
data_folder = root_folder / 'data'
input_folder = data_folder / 'input'
output_folder = data_folder / 'output'

log_folder = input_folder / 'logs'
pn_folder = input_folder / 'petrinets'

epochs = 300
folds = 1
train_ratio = 0.7
variant_split = 0.9
validation_split = 0.2

log_list = [
    # 'sepsis_cases_1.csv'
    # 'helpdesk.csv' 
    # 'BPIC12.csv' 
    # 'Road_Traffic.csv'
    # 'BPIC13_I.csv'
    # 'BPIC13_CP.csv',
    'Synthetic.xes'
]
