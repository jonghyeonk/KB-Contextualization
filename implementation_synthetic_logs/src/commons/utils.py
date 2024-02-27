import glob
import os
from pathlib import Path
from src.commons import shared_variables as shared
from src.commons.log_utils import LogData


def extract_last_model_checkpoint(log_name: str, models_folder: str, fold: int, model_type: str) -> Path:
    model_filepath = shared.output_folder / models_folder / str(fold) / 'models' / model_type / log_name
    list_of_files = glob.glob(str(model_filepath / '*.tf'))
    latest_file = max(list_of_files, key=os.path.getctime)
    return Path(latest_file)


def extract_bk_filename(log_name: str) -> Path:
    if log_name.endswith('UBE') or log_name.endswith('Synthetic') or  log_name.endswith('card') or log_name.endswith('mccloud'):
        name = shared.pn_folder / (log_name + '.bpmn')
    else:
        name = shared.pn_folder / (log_name + '.pnml')

    return name

def extract_trace_sequences(log_data: LogData, trace_ids: list, resource: bool, outcome: bool) -> list:
    """
    Extract activity, resource and output sequences starting from a list of trace ids (i.e. trace_names).
    """
    act_seqs = []  # list of all the activity sequences
    res_seqs = []  # list of all the resource sequences
    outcomes = []  # outcome of each sequence (i.e. each case)
    
    traces = log_data.log[log_data.log[log_data.case_name_key].isin(trace_ids)]
    for _, trace in traces.groupby(log_data.case_name_key):
        line = ''.join(trace[log_data.act_name_key].tolist())  # sequence of activities for one case
        act_seqs.append(line)

        if resource:
            line_group = ''.join(trace[log_data.res_name_key].tolist())  # sequence of groups for one case
            res_seqs.append(line_group)
        if outcome:
            o = trace[log_data.label_name_key].iloc[0]
            outcomes.append(o)

    return act_seqs, res_seqs,  outcomes
