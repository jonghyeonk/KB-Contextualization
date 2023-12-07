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
train_ratio = 0.7
variant_split = 0.9
validation_split = 0.2

log_list = [
    #  'Synthetic log labelled.xes'
    # 'Production.csv'
    # 'sepsis_cases_1.csv'
    # 'sepsis_cases_1_sampled10.csv',
    # 'sepsis_cases_1_sampled20.csv',
    # 'sepsis_cases_1_sampled30.csv',
    # 'sepsis_cases_1_sampled40.csv',
    # 'sepsis_cases_1_sampled50.csv'
    # 'sepsis_cases_2.csv'
#     'sepsis_cases_4.csv',
    # 'Synthetic_UBE.csv',
    # 'credit_card.csv',
    # 'mccloud.csv',
    # 'hospital_billing_2.csv',
    # 'hospital_billing_3.csv'
    # 'helpdesk.csv'  #O
    'BPIC12.csv' #O
    # 'Road_Traffic.csv'
    # 'BPIC13_I.csv'
    # 'BPIC13_CP.csv'
]
