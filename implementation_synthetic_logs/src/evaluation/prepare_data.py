"""
This script prepares data in the format for the testing
algorithms to run
The script is expanded to the resource attribute
"""

from __future__ import division
import copy
import re
from pathlib import Path
from typing import Dict

import numpy as np
import pm4py
import pandas as pd

from src.commons.log_utils import LogData

def prepare_testing_data(log_data: LogData,  training_traces: pd.DataFrame, resource: bool):
    """
    Get all possible symbols for activities and resources and annotate them with integers.
    """
    act_name_key = log_data.act_name_key
    
    act_chars = list(training_traces[act_name_key].unique())
    act_chars.sort()
    
        
    check_new_act = log_data.log[act_name_key].unique().tolist()
    
    act_chars2 = act_chars + [na for na in check_new_act if na not in act_chars]

    target_act_chars = copy.copy(act_chars)
    target_act_chars.append('!')
    target_act_chars.sort()

    target_act_chars2 = copy.copy(act_chars2)
    target_act_chars2.append('!')
    target_act_chars2.sort()   

    act_to_int = dict((c, i) for i, c in enumerate(act_chars))
    target_act_to_int = dict((c, i) for i, c in enumerate(target_act_chars))
    target_int_to_act = dict((i, c) for i, c in enumerate(target_act_chars))

    target_act_to_int2 = dict((c, i) for i, c in enumerate(target_act_chars2))
    target_int_to_act2 = dict((i, c) for i, c in enumerate(target_act_chars2))

    if resource:
        res_name_key = log_data.res_name_key
        res_chars = list(training_traces[res_name_key].unique())
        res_chars.sort()
        
        check_new_res = log_data.log[res_name_key].unique()
        if len(check_new_res) > len(res_chars):
            print("New resource name unfound in train_set exists in test set")
            res_chars = res_chars + [nr for nr in check_new_res if nr not in res_chars]

        target_res_chars = copy.copy(res_chars)
        target_res_chars.append('!')
        target_res_chars.sort()

        res_to_int = dict((c, i) for i, c in enumerate(res_chars))
        target_res_to_int = dict((c, i) for i, c in enumerate(target_res_chars))
        target_int_to_res = dict((i, c) for i, c in enumerate(target_res_chars))
    else:
        res_to_int = None
        target_res_to_int = None
        target_int_to_res = None

    return act_to_int, target_act_to_int, target_int_to_act, target_act_to_int2, target_int_to_act2, res_to_int, target_res_to_int, target_int_to_res


def select_petrinet_compliant_traces(log_data: LogData,  method_fitness: str, traces: pd.DataFrame, path_to_pn_model_file: Path):
    """
    Select traces compliant to a Petri Net at least in a certain percentage specified as compliance_th
    """

    compliant_trace_ids = []
    if (method_fitness == "fitness_alignments") or  (method_fitness == "conformance_diagnostics_alignments_prefix"):
        method_fitness = "conformance_diagnostics_alignments"
    elif method_fitness == "fitness_token_based_replay":
        method_fitness = "conformance_diagnostics_token_based_replay"

    for trace_id, fitness in get_pn_fitness(path_to_pn_model_file, method_fitness, traces, log_data).items():
        if fitness >= log_data.compliance_th:
            compliant_trace_ids.append(trace_id)

    compliant_traces = traces[traces[log_data.case_name_key].isin(compliant_trace_ids)]
    return compliant_traces


def get_pn_fitness(bk_file: Path, method_fitness: str, log: pd.DataFrame, log_data: LogData) -> Dict[str, float]:
    # Decode traces for feeding them to the Petri net
    dec_log = log.replace(to_replace={
        log_data.act_name_key: log_data.act_enc_mapping,
    })

    dec_log[log_data.timestamp_key] = pd.to_datetime(log_data.log[log_data.timestamp_key], unit='s') 

    if 'bpmn' in str(bk_file):
        bpmn = pm4py.read_bpmn(str(bk_file))
        net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn)
    else:
        net, initial_marking, final_marking = pm4py.read_pnml(str(bk_file))


    if method_fitness == "conformance_diagnostics_alignments":
        alignments = pm4py.conformance_diagnostics_alignments(dec_log,  net, initial_marking, final_marking,
                                                            activity_key=log_data.act_name_key,
                                                            case_id_key=log_data.case_name_key,
                                                            timestamp_key= log_data.timestamp_key)
        trace_fitnesses = [a['fitness'] for a in alignments]

        
    elif method_fitness == "fitness_alignments":
        alignments = pm4py.fitness_alignments(dec_log,  net, initial_marking, final_marking,
                                                          activity_key=log_data.act_name_key,
                                                          case_id_key=log_data.case_name_key,
                                                          timestamp_key= log_data.timestamp_key)
                                                          
        trace_fitnesses = [alignments['log_fitness']]
    
    elif method_fitness == "conformance_diagnostics_token_based_replay":
        alignments = pm4py.conformance_diagnostics_token_based_replay(dec_log,  net, initial_marking, final_marking,
                                                          activity_key=log_data.act_name_key,
                                                          case_id_key=log_data.case_name_key,
                                                          timestamp_key= log_data.timestamp_key)
        trace_fitnesses = [a['trace_fitness'] for a in alignments]
        
    elif method_fitness == "fitness_token_based_replay":
        alignments = pm4py.fitness_token_based_replay(dec_log,  net, initial_marking, final_marking,
                                                          activity_key=log_data.act_name_key,
                                                          case_id_key=log_data.case_name_key,
                                                          timestamp_key= log_data.timestamp_key)
        trace_fitnesses = [alignments['log_fitness']]
        
        
    trace_ids = list(log[log_data.case_name_key].unique())
    trace_ids_with_fitness = dict(zip(trace_ids, trace_fitnesses))
    return trace_ids_with_fitness





# === Helper functions ===


def encode(crop_trace: pd.DataFrame, log_data: LogData,  maxlen: int, char_indices: Dict[str, int],
                      char_indices_group: Dict[str, int], resource: bool) -> np.ndarray:
    """
    Onehot encoding of an ongoing trace (control-flow + resource)
    """
    chars = list(char_indices.keys())
    
    if resource:
        sentence = ''.join(crop_trace[log_data.act_name_key].tolist())
        sentence_group = ''.join(crop_trace[log_data.res_name_key].tolist())
        chars_group = list(char_indices_group.keys())
        num_features = len(chars) + len(chars_group) + 1
        x = np.zeros((1, maxlen, num_features), dtype=np.float32)
        leftpad = maxlen - len(sentence)
        for t, char in enumerate(sentence):
            for c in chars:
                if c == char:
                    x[0, t + leftpad, char_indices[c]] = 1
            for g in chars_group:
                if g == sentence_group[t]:
                    x[0, t + leftpad, len(char_indices) + char_indices_group[g]] = 1

            x[0, t + leftpad, len(chars) + len(chars_group)] = t + 1
    else:
        sentence = ''.join(crop_trace[log_data.act_name_key].tolist())
        num_features = len(chars) + 1 
        x = np.zeros((1, maxlen, num_features), dtype=np.float32)
        leftpad = maxlen - len(sentence) 
        for t, char in enumerate(sentence):
            for c in chars:
                if c == char:
                    x[0, t + leftpad, char_indices[c]] = 1
            x[0, t + leftpad, len(chars)] = t + 1  
            
    return x


def repetitions(seq: str):
    r = re.compile(r"(.+?)\1+")
    for match in r.finditer(seq):
        indices = [index + match.start()
                   for index in range(len(seq[match.start(): match.end()]))
                   if seq[match.start(): match.end()].startswith(match.group(1), index)]
        yield match.group(1), len(match.group(0)) / len(match.group(1)), indices


def reduce_act_loop_probability(act_seq):
    tmp = dict()

    # loop_len = number of consequent repetitions of loop inside trace
    for loop, loop_len, _ in repetitions(act_seq):
        if act_seq.endswith(loop):
            loop_start_symbol = loop[0]
            reduction_factor = 1 / np.math.exp(loop_len)

            if loop_start_symbol in tmp:
                if reduction_factor > tmp[loop_start_symbol]:
                    tmp[loop_start_symbol] = reduction_factor
            else:
                tmp[loop_start_symbol] = reduction_factor

    for loop_start_symbol, reduction_factor in tmp.items():
        yield loop_start_symbol, reduction_factor


def get_act_prediction(prefix, prediction, target_ind_to_act, target_act_to_ind, reduce_loop_prob=True, ith_best=0):
    if reduce_loop_prob:
        for symbol_where_loop_starts, reduction_factor in reduce_act_loop_probability(prefix):
            # Reducing probability of the first symbol of detected loop (if any) for preventing endless traces
            symbol_idx = target_act_to_ind[symbol_where_loop_starts]
            prediction[symbol_idx] *= reduction_factor
    
    pred_idx = np.argsort(prediction)[len(prediction) - ith_best - 1]
    return target_ind_to_act[pred_idx]



def get_beam_size(self, NodePrediction, current_prediction_premis, prefix, prefix_trace, prediction, res_prediction, y_char, fitness, act_ground_truth_org,
                  char_indices, target_ind_to_act, target_act_to_ind, target_ind_to_res, target_res_to_ind, step, 
                  log_data, resource, beam_size):
    
    record = []
    if resource:
        act_prefix = prefix.cropped_line
        res_prefix = prefix.cropped_line_group
        for j in range(beam_size):
            temp_prediction = get_predictions(act_prefix, res_prefix, prediction, res_prediction,
                                            target_ind_to_act, target_act_to_ind, target_ind_to_res, target_res_to_ind, reduce_loop_prob=True,ith_best=j)
            
            predicted_row = prefix_trace.tail(1).copy()
            predicted_row.loc[:, log_data.act_name_key] = temp_prediction
            temp_cropped_trace_next = pd.concat([prefix_trace, predicted_row], axis= 0)
            probability_this = np.sort(prediction)[len(prediction) - 1 - j]

            temp = NodePrediction(temp_cropped_trace_next,
                                    current_prediction_premis.probability_of + np.log(probability_this))
            self.put(temp)  
    else:
        for j in range(beam_size):
            temp_prediction = get_act_prediction(prefix, prediction, target_ind_to_act, target_act_to_ind, reduce_loop_prob=True, ith_best=j)
            predicted_row = prefix_trace.tail(1).copy()
            predicted_row.loc[:, log_data.act_name_key] = temp_prediction
            temp_cropped_trace_next = pd.concat([prefix_trace, predicted_row], axis= 0)
            probability_this = np.sort(prediction)[len(prediction) - 1 - j]
            
            temp = NodePrediction(temp_cropped_trace_next,
                                    current_prediction_premis.probability_of + np.log(probability_this))            

            trace_org = [log_data.act_enc_mapping[i] if i != "!" else "" for i in temp_cropped_trace_next[log_data.act_name_key].tolist()]
            
            if len(fitness) > 0 :
                fitness_sorted = np.array(fitness)[np.argsort(prediction)]
                fitness_this = fitness_sorted[len(fitness_sorted) - 1 - j]
                y_char_sorted = np.array(y_char)[np.argsort(prediction)]
                y_char_this = y_char_sorted[len(y_char_sorted) - 1 - j]

                record.append(str(
                    "trace_org = " + '>>'.join(  trace_org ) + 
                    "// previous = " + str(round( current_prediction_premis.probability_of, 3)) +
                    "// current = " + str(round( current_prediction_premis.probability_of + np.log(probability_this), 3)) +
                    "// rnn = " + str(round(y_char_this,3)) +
                    "// fitness = " + str(round(fitness_this,3))) +
                    "&"
                    )        

            self.put(temp)    
               
    return self, record


def reduce_loop_probability(act_seq, res_seq):
    tmp = dict()

    for loop, loop_len, loop_start_indices in repetitions(act_seq):
        if act_seq.endswith(loop):
            loop_start_symbol = loop[0]
            reduction_factor = 1 / np.math.exp(loop_len)
            loop_related_resources = set(res_seq[i] for i in loop_start_indices)

            if loop_start_symbol in tmp:
                tmp[loop_start_symbol][0].update(loop_related_resources)
                if reduction_factor > tmp[loop_start_symbol][1]:
                    tmp[loop_start_symbol][1] = reduction_factor
            else:
                tmp[loop_start_symbol] = [loop_related_resources, reduction_factor]

    for loop_start_symbol, lst in tmp.items():
        yield loop_start_symbol, lst[0], lst[1]


def get_predictions(act_prefix, res_prefix, act_pred, res_pred, target_ind_to_act, target_act_to_ind, target_ind_to_res,
                    target_res_to_ind, reduce_loop_prob=True, ith_best=0):
    if reduce_loop_prob:
        for start_act_of_loop, related_res_list, reduction_factor in reduce_loop_probability(act_prefix, res_prefix):
            # Reducing probability of the first symbol of detected loop (if any) and of the related resources
            # for preventing endless traces
            act_idx = target_act_to_ind[start_act_of_loop]
            act_pred[act_idx] *= reduction_factor

            for res in related_res_list:
                res_idx = target_res_to_ind[res]
                res_pred[res_idx] *= reduction_factor

    act_pred_idx = np.argsort(act_pred)[len(act_pred) - ith_best - 1]
    res_pred_idx = np.argsort(res_pred)[len(res_pred) - ith_best - 1]

    return target_ind_to_act[act_pred_idx], target_ind_to_res[res_pred_idx]
