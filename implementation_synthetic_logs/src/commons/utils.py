import glob
import os
from pathlib import Path
import time #jh
from src.commons import shared_variables as shared
from src.commons.log_utils import LogData


def extract_last_model_checkpoint(log_name: str, models_folder: str, fold: int, model_type: str) -> Path:
    model_filepath = shared.output_folder / models_folder / str(fold) / 'models' / model_type / log_name
    list_of_files = glob.glob(str(model_filepath / '*.h5'))
    latest_file = max(list_of_files, key=os.path.getctime)
    return Path(latest_file)


def extract_petrinet_filename(log_name: str) -> Path:
    return shared.pn_folder / (log_name + '.pnml')

def extract_tree_filename(log_name: str) -> Path:
    return shared.pn_folder / (log_name + '.ptml')

def extract_trace_sequences(log_data: LogData, trace_ids: [str]) -> ([str], [str], [float], [float], [float], [float], [str]):
    """
    Extract activity, resource and output sequences starting from a list of trace ids (i.e. trace_names).
    """
    act_seqs = []  # list of all the activity sequences
    time_seqs = [] #jh
    time_seqs2 = [] #jh
    time_seqs3 = [] #jh
    time_seqs4 = [] #jh
    traces = log_data.log[log_data.log[log_data.case_name_key].isin(trace_ids)]
    for _, trace in traces.groupby(log_data.case_name_key):
        line = ''.join(trace[log_data.act_name_key].tolist())  # sequence of activities for one case
        act_seqs.append(line)

        # line_time = trace[log_data.timestamp_key].tolist() + [0]    #jh
        # line_time2 = trace[log_data.timestamp_key2].tolist() + [0]    #jh
        # line_time3 = trace[log_data.timestamp_key3].tolist() + [0]    #jh
        # line_time4 = trace[log_data.timestamp_key4].tolist() + [0]    #jh
        # time_seqs.append(line_time)
        # time_seqs2.append(line_time2)
        # time_seqs3.append(line_time3)
        # time_seqs4.append(line_time4)


    return act_seqs
