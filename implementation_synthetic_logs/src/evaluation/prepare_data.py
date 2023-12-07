"""
This script prepares data in the format for the testing
algorithms to run
The script is expanded to the resource attribute
"""

from __future__ import division
import copy
import re
from pathlib import Path

import numpy as np
import pm4py
import pandas as pd

from src.commons.log_utils import LogData
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.objects.conversion.process_tree import converter as process_tree_converter
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness


def prepare_testing_data(log_data: LogData, training_traces: pd.DataFrame):
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


    return act_to_int, target_act_to_int, target_int_to_act, target_act_to_int2, target_int_to_act2


def select_petrinet_compliant_traces(log_data: LogData, traces: pd.DataFrame, path_to_pn_model_file: Path):
    """
    Select traces compliant to a Petri Net at least in a certain percentage specified as compliance_th
    """
    compliant_trace_ids = []
    for trace_id, fitness in get_token_fitness(path_to_pn_model_file, traces, log_data).items():
        if fitness >= log_data.compliance_th:
            compliant_trace_ids.append(trace_id)

    compliant_traces = traces[traces[log_data.case_name_key].isin(compliant_trace_ids)]
    return compliant_traces


def get_token_fitness(pn_file: Path, log: pd.DataFrame, log_data: LogData) -> dict[str: float]:
    # Decode traces for feeding them to the Petri net
    dec_log = log.replace(to_replace={
        log_data.act_name_key: log_data.act_enc_mapping,
    })

    # dec_log['time:timestamp'] =  pd.to_datetime(log_data.log['Complete Timestamp']) # for product data
    dec_log[log_data.timestamp_key] = pd.to_datetime(log_data.log[log_data.timestamp_key], unit='s') #jh

    bpmn = pm4py.read_bpmn('Synthetic.bpmn')
    # net, initial_marking, final_marking = pm4py.read_pnml(str(pn_file))

    net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn)
    
    # pm4py.fitness_token_based_replay
    # pm4py.conformance_diagnostics_token_based_replay
    alignments = pm4py.conformance_diagnostics_token_based_replay(dec_log,  net, initial_marking, final_marking,
                                                          activity_key=log_data.act_name_key,
                                                        #   timestamp_key= 'time:timestamp',
                                                          case_id_key=log_data.case_name_key)

    trace_ids = list(log[log_data.case_name_key].unique())
    trace_fitnesses = [a['trace_fitness'] for a in alignments]
    trace_ids_with_fitness = dict(zip(trace_ids, trace_fitnesses))
    return trace_ids_with_fitness


def get_token_fitness2(pn_file: Path, log: pd.DataFrame, log_data: LogData) -> dict[str: float]:
    # Decode traces for feeding them to the Petri net
    dec_log = log.replace(to_replace={
        log_data.act_name_key: log_data.act_enc_mapping,
    })

    # dec_log['time:timestamp'] =  pd.to_datetime(log_data.log['Complete Timestamp']) # for product data
    dec_log[log_data.timestamp_key] = pd.to_datetime(log_data.log[log_data.timestamp_key], unit='s') #jh

    
    bpmn = pm4py.read_bpmn('Synthetic.bpmn')
    # net, initial_marking, final_marking = pm4py.read_pnml(str(pn_file))

    net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn)
    
    # pm4py.fitness_token_based_replay
    # pm4py.conformance_diagnostics_token_based_replay
    alignments = pm4py.fitness_token_based_replay(dec_log,  net, initial_marking, final_marking,
                                                          activity_key=log_data.act_name_key,
                                                        #   timestamp_key= 'time:timestamp',
                                                          case_id_key=log_data.case_name_key)

    trace_ids = list(log[log_data.case_name_key].unique())
    trace_fitnesses = [alignments['log_fitness']]
    trace_ids_with_fitness = dict(zip(trace_ids, trace_fitnesses))
    return trace_ids_with_fitness

def get_pn_fitness(pn_file: Path, log: pd.DataFrame, log_data: LogData) -> dict[str: float]:
    # Decode traces for feeding them to the Petri net
    dec_log = log.replace(to_replace={
        log_data.act_name_key: log_data.act_enc_mapping
    })

    # dec_log['time:timestamp'] =  pd.to_datetime(log_data.log['Complete Timestamp']) # for product data
    dec_log[log_data.timestamp_key] = pd.to_datetime(log_data.log[log_data.timestamp_key], unit='s') #jh

    bpmn = pm4py.read_bpmn('Synthetic.bpmn')
    # net, initial_marking, final_marking = pm4py.read_pnml(str(pn_file))
    alignments = pm4py.conformance_diagnostics_alignments(dec_log,  bpmn,
                                                          activity_key=log_data.act_name_key,
                                                        #   timestamp_key= 'time:timestamp',
                                                          case_id_key=log_data.case_name_key)
    
    trace_ids = list(log[log_data.case_name_key].unique())
    trace_fitnesses = [a['fitness'] for a in alignments]
    
    trace_ids_with_fitness = dict(zip(trace_ids, trace_fitnesses))
    return trace_ids_with_fitness

def get_pn_fitness2(pn_file: Path, log: pd.DataFrame, log_data: LogData) -> dict[str: float]:
    # Decode traces for feeding them to the Petri net
    dec_log = log.replace(to_replace={
        log_data.act_name_key: log_data.act_enc_mapping,
        log_data.res_name_key: log_data.res_enc_mapping
    })

    # dec_log['time:timestamp'] =  pd.to_datetime(log_data.log['Complete Timestamp']) # for product data
    dec_log[log_data.timestamp_key] = pd.to_datetime(log_data.log[log_data.timestamp_key], unit='s') #jh

    
    net, initial_marking, final_marking = pm4py.read_pnml(str(pn_file))

    alignments = pm4py.fitness_alignments(dec_log,  net, initial_marking, final_marking,
                                                          activity_key=log_data.act_name_key,
                                                        #   timestamp_key= 'time:timestamp',
                                                          case_id_key=log_data.case_name_key)

    trace_ids = list(log[log_data.case_name_key].unique())
    trace_fitnesses = [alignments['log_fitness']]
    
    trace_ids_with_fitness = dict(zip(trace_ids, trace_fitnesses))
    return trace_ids_with_fitness


def get_tr_fitness(pt_file: Path, log: pd.DataFrame, log_data: LogData, trace_name) -> float:
    # Decode traces for feeding them to the Petri net
    dec_log = log.replace(to_replace={
        log_data.act_name_key: log_data.act_enc_mapping,
        log_data.res_name_key: log_data.res_enc_mapping
    })
    dec_log[log_data.timestamp_key] = pd.to_datetime(log_data.log[log_data.timestamp_key], unit='s') #jh

    tree = pm4py.read_ptml(str(pt_file))

    # tree = pm4py.convert_to_process_tree(net, initial_marking, final_marking)
    net, initial_marking, final_marking = process_tree_converter.apply(tree)

    dec_log[log_data.case_name_key] = trace_name 
    dec_log = pm4py.format_dataframe(dec_log, 
                                     case_id=log_data.case_name_key, 
                                     activity_key=log_data.act_name_key, 
                                     timestamp_key=log_data.timestamp_key)

    # event_log = pm4py.convert_to_event_log(dec_log)
    aligned_traces = alignments.apply_log(dec_log, net, initial_marking, final_marking)
                                        #   activity_key=log_data.act_name_key,
                                    #   timestamp_key= 'time:timestamp',
                                        # case_id_key=log_data.case_name_key)

    trace_ids_with_fitness = replay_fitness.evaluate(aligned_traces, variant=replay_fitness.Variants.ALIGNMENT_BASED)

    
    return list(trace_ids_with_fitness.values())[3]


# === Helper functions ===

def encode(sentence: str, maxlen: int, char_indices: dict[str, int]) -> np.ndarray:
    """
    Onehot encoding of an ongoing trace (only control-flow)
    """
    chars = list(char_indices.keys())
    num_features = len(chars) + 1
    x = np.zeros((1, maxlen, num_features), dtype=np.float32)
    leftpad = maxlen - len(sentence)
    for t, char in enumerate(sentence):
        for c in chars:
            if c == char:
                x[0, t + leftpad, char_indices[c]] = 1
        x[0, t + leftpad, len(chars)] = t + 1
    return x




def encode_with_group0(sentence: str, sentence_group: str,  maxlen: int, char_indices: dict[str, int],
                      char_indices_group: dict[str, int]) -> np.ndarray:
    """
    Onehot encoding of an ongoing trace (control-flow + resource)
    """
    chars = list(char_indices.keys())
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
    return x


def encode_with_group(sentence: str, sentence_group: str, sentence_time: list, maxlen: int, char_indices: dict[str, int],
                      char_indices_group: dict[str, int]) -> np.ndarray:
    """
    Onehot encoding of an ongoing trace (control-flow + resource)
    """
    chars = list(char_indices.keys())
    chars_group = list(char_indices_group.keys())
    num_features = len(chars) + len(chars_group) + 2
    x = np.zeros((1, maxlen, num_features), dtype=np.float32)
    leftpad = maxlen - len(sentence)
    for t, char in enumerate(sentence):
        for c in chars:
            if c == char:
                x[0, t + leftpad, char_indices[c]] = 1
        for g in chars_group:
            if g == sentence_group[t]:
                x[0, t + leftpad, len(char_indices) + char_indices_group[g]] = 1
        for ti in sentence_time:
                x[0, t + leftpad, len(chars) + len(chars_group) ] = ti    
        
        x[0, t + leftpad, len(chars) + len(chars_group)+1] = t + 1
    return x


def encode_with_group2(sentence: str, sentence_group: str, sentence_time: list, sentence_time2: list, 
                       sentence_time3: list, sentence_time4: list, maxlen: int, char_indices: dict[str, int],
                      char_indices_group: dict[str, int]) -> np.ndarray:  #jh
    """
    Onehot encoding of an ongoing trace (control-flow + resource)
    """
    chars = list(char_indices.keys())
    chars_group = list(char_indices_group.keys())
    num_features = len(chars) + len(chars_group) + 5
    x = np.zeros((1, maxlen, num_features), dtype=np.float32)
    leftpad = maxlen - len(sentence)
    for t, char in enumerate(sentence):
        for c in chars:
            if c == char:
                x[0, t + leftpad, char_indices[c]] = 1
        for g in chars_group:
            if g == sentence_group[t]:
                x[0, t + leftpad, len(char_indices) + char_indices_group[g]] = 1
        for ti in sentence_time:
                x[0, t + leftpad, len(chars) + len(chars_group) ] = ti    
        for ti in sentence_time2:
                x[0, t + leftpad, len(chars) + len(chars_group)+1 ] = ti     
        for ti in sentence_time3:
                x[0, t + leftpad, len(chars) + len(chars_group)+2 ] = ti  
        for ti in sentence_time4:
                x[0, t + leftpad, len(chars) + len(chars_group)+3 ] = ti                       
        x[0, t + leftpad, len(chars) + len(chars_group)+4] = t + 1
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



def get_beam_size(self, NodePrediction, current_prediction_premis, prefix, prefix_trace, prediction,y_char,fitness,act_ground_truth_org,
                  target_ind_to_act, target_act_to_ind, log_data, beam_size):
    
    for j in range(beam_size):
        temp_prediction = get_act_prediction(prefix, prediction, target_ind_to_act, target_act_to_ind, reduce_loop_prob=True, ith_best=j)
        predicted_row = prefix_trace.tail(1).copy()
        predicted_row.loc[:, log_data.act_name_key] = temp_prediction
        temp_cropped_trace_next = pd.concat([prefix_trace, predicted_row], axis= 0)
        probability_this = np.sort(prediction)[len(prediction) - 1 - j]
        # fitness_sorted = np.array(fitness)[np.argsort(prediction)]
        # fitness_this = fitness_sorted[len(fitness_sorted) - 1 - j]
        # y_char_sorted = np.array(y_char)[np.argsort(prediction)]
        # y_char_this = y_char_sorted[len(y_char_sorted) - 1 - j]
        temp = NodePrediction(temp_cropped_trace_next,
                                current_prediction_premis.probability_of + np.log(probability_this)) # current_prediction_premis.probability_of + np.log(probability_this) # probability_this
        trace_org = [log_data.act_enc_mapping[i] if i != "!" else "" for i in temp_cropped_trace_next[log_data.act_name_key].tolist()]
        
        # if len( [i for i in act_ground_truth_org if i in ['Unexpected2', 'Repairing2', 'Unexpected1', 'Repairing1'] ] ) > 0:
            
        #     print(
        #         #"trace = ", ''.join(temp_cropped_trace_next[log_data.act_name_key].tolist()) , 
        #         "trace_org = ", '>>'.join(  trace_org ) , 
        #         ", previous = ", round( current_prediction_premis.probability_of, 3),
        #         ", current = ", round( current_prediction_premis.probability_of + np.log(probability_this), 3),
        #         ", rnn = ", round(y_char_this,3),
        #         ", fitness = ", round(fitness_this,3)
        #         )     
        
        self.put(temp)
        
    return self


def get_beam_size2(self, NodePrediction, current_prediction_premis, prefix, prefix_trace, prediction,y_char, fitness,
                   target_ind_to_act, target_act_to_ind, log_data, beam_size):
    
    for j in range(beam_size):
        temp_prediction = get_act_prediction(prefix, prediction, target_ind_to_act, target_act_to_ind, reduce_loop_prob=True, ith_best=j)
        predicted_row = prefix_trace.tail(1).copy()
        predicted_row.loc[:, log_data.act_name_key] = temp_prediction
        temp_cropped_trace_next = pd.concat([prefix_trace, predicted_row], axis= 0)
        probability_this = np.sort(prediction)[len(prediction) - 1 - j]
        fitness_sorted = np.array(fitness)[np.argsort(prediction)]
        fitness_this = fitness_sorted[len(fitness_sorted) - 1 - j]
        y_char_sorted = np.array(y_char)[np.argsort(prediction)]
        y_char_this = y_char_sorted[len(y_char_sorted) - 1 - j]
        temp = NodePrediction(temp_cropped_trace_next,
                                current_prediction_premis.probability_of + np.log(probability_this))
        # print("trace = ", ''.join(temp_cropped_trace_next[log_data.act_name_key].tolist()) , 
        #       ", previous = ", round( current_prediction_premis.probability_of, 3),
        #       ", current = ", round( current_prediction_premis.probability_of + np.log(probability_this), 3),
        #       ", rnn = ", round(y_char_this,3),
        #       ", fitness = ", round(fitness_this,3)
        #       )
        
        self.put(temp)
        
    return self


def reduce_loop_probability(act_seq, res_seq):
    tmp = dict()

    # loop_len = number of consequent repetitions of loop inside trace
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
