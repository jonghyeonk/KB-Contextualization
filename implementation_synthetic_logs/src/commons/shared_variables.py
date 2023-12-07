"""
This file was created in order to bring
common variables and functions into one file to make
code more clear
"""
from pathlib import Path


ascii_offset = 161
beam_size = 3
th_reduction_factor = 1

data_folder = Path.cwd() / 'data'
input_folder = data_folder / 'input'
output_folder = data_folder / 'output'

log_folder = input_folder / 'logs'
pn_folder = input_folder / 'petrinets'

epochs = 300
folds = 3
validation_split = 0.2

log_list = [
    'Synthetic.xes'
    #  'Synthetic log labelled.xes'
    # 'Production.csv'
    #'sepsis_cases_1.csv'
    # 'sepsis_cases_2.csv'
#     'sepsis_cases_4.csv'
]
