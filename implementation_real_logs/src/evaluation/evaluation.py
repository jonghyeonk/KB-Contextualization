import time
import pm4py
import itertools
from pathlib import Path

from src.commons.log_utils import LogData
from src.commons import shared_variables as shared
from src.commons.utils import extract_bk_filename, extract_last_model_checkpoint
from src.evaluation.prepare_data import prepare_testing_data, select_petrinet_compliant_traces
from src.evaluation.inference_algorithms import beamsearch_cf

from pm4py.algo.simulation.playout.petri_net import algorithm
from pm4py.algo.simulation.playout.petri_net.variants import extensive
from pm4py.algo.simulation.playout.petri_net.variants.extensive import Parameters
from pm4py.statistics.variants.pandas import get as variants_get
from pm4py.utils import get_properties

def evaluate_all(log_data: LogData, models_folder: str, alg: str, method_fitness: str, weight: list, resource: bool, timestamp: bool, outcome: bool):
    start_time = time.time()

    training_traces = log_data.log[log_data.log[log_data.case_name_key].isin(log_data.training_trace_ids)]
    # +1 accounts of fake '!' symbol added to identify end of trace
    maxlen = max([trace.shape[0] for _, trace in training_traces.groupby(log_data.case_name_key)]) + 1
    predict_size = maxlen
    print(maxlen)
    act_to_int, target_act_to_int, target_int_to_act, target_act_to_int2, target_int_to_act2, res_to_int, target_res_to_int, target_int_to_res \
        = prepare_testing_data(log_data, training_traces, resource)

    check_new_act = log_data.log[log_data.act_name_key].unique().tolist()
    new_chars = [na for na in check_new_act if na not in target_act_to_int]
    log_data.new_chars = new_chars
    
    
    bk_filename = extract_bk_filename(log_data.log_name.value, log_data.evaluation_prefix_start)
    
    if (method_fitness == "conformance_diagnostics_alignments_prefix"):
        if 'bpmn' in str(bk_filename):
            bpmn = pm4py.read_bpmn(str(bk_filename))
            net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn)
            pm4py.write_pnml(net, initial_marking, final_marking, str(bk_filename).split(".")[0] + ".pnml")

        else:
            net, initial_marking, final_marking = pm4py.read_pnml(bk_filename)
            
        print("Start unfolding petrinet")
        sim_log = algorithm.apply(net, initial_marking, final_marking=final_marking, variant=extensive, parameters= {Parameters.MAX_TRACE_LENGTH: min(30, maxlen) })
        sim_data = pm4py.convert_to_dataframe(sim_log)
        print("Finished unfolding petrinet")
        
        # print(sim_data.head())
        # print(len(sim_data['case:concept:name'].unique()))

        # parameters = get_properties(sim_data, case_id_key ='case:concept:name', activity_key = 'concept:name', timestamp_key = 'time:timestamp')
        # variants = variants_get.get_variants_count(sim_data, parameters=parameters)
        # print(len(variants))

        prefix = list(sim_data.groupby('case:concept:name').apply(lambda x: list(range(1, len(x)+1))))
        prefix = list(itertools.chain(*prefix))
        sim_data['prefix'] = prefix
        
        for prefix_len in range(log_data.evaluation_prefix_start, predict_size+1):
            sim_data_prefix = sim_data.loc[sim_data['prefix'] < prefix_len+1].reset_index(drop= True)
            net_prefix, im_prefix, fm_prefix = pm4py.discover_petri_net_inductive(sim_data_prefix)
            pm4py.write_pnml(net_prefix, im_prefix, fm_prefix, str(bk_filename).split(".")[0] + "_" + str(prefix_len) + ".pnml")
    
    evaluation_traces =  log_data.log[log_data.log[log_data.case_name_key].isin(log_data.evaluation_trace_ids)]
    
    # dec_log = evaluation_traces.replace(to_replace={
    # log_data.act_name_key: log_data.act_enc_mapping})
    
    
    # print(evaluation_traces.head())
    # evaluation_traces =  log_data.test_log
    # print(bk_filename)
    # print(dec_log.head())
    # net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(dec_log, noise_threshold = 0.30,  activity_key=log_data.act_name_key,
    #                                                     case_id_key=log_data.case_name_key,
    #                                                     timestamp_key= log_data.timestamp_key)    
    
    
    # net, initial_marking, final_marking = pm4py.discover_petri_net_alpha(dec_log, activity_key=log_data.act_name_key,
    #                                                     case_id_key=log_data.case_name_key,
    #                                                     timestamp_key= log_data.timestamp_key)      

    # net, initial_marking, final_marking = pm4py.discover_petri_net_heuristics(dec_log, activity_key=log_data.act_name_key,
    #                                                     case_id_key=log_data.case_name_key,
    #                                                     timestamp_key= log_data.timestamp_key) 
    
    
    tree  = pm4py.discover_process_tree_inductive(evaluation_traces, noise_threshold = 0.45,  activity_key=log_data.act_name_key,
                                                        case_id_key=log_data.case_name_key,
                                                        timestamp_key= log_data.timestamp_key)    
    net, initial_marking, final_marking = pm4py.convert_to_petri_net(tree)
    
    
    pm4py.write_pnml(net, initial_marking, final_marking, bk_filename)
    # compliant_traces = select_petrinet_compliant_traces(log_data, method_fitness, evaluation_traces, bk_filename)
    compliant_traces = evaluation_traces
    
            
            
    print("Compliant traces: " + str(compliant_traces[log_data.case_name_key].nunique())
          + " out of " + str(len(log_data.evaluation_trace_ids)))
    print('Elapsed time:', time.time() - start_time)

    for fold in range(shared.folds):
        eval_algorithm = alg + "_cf" + "r"*resource + "t"*timestamp + "o"*outcome
        start_time = time.time()

        # algorithm_name = Path(eval_algorithm.__file__).stem
        folder_path = shared.output_folder / models_folder / str(fold) / 'results' / eval_algorithm
        if not Path.exists(folder_path):
            Path.mkdir(folder_path, parents=True)
        output_filename = folder_path / f'{log_data.log_name.value}_beam{str(shared.beam_size)}_fold{str(fold)}_cluster{log_data.evaluation_prefix_start}.csv' 

        print(f"fold {fold} - {eval_algorithm}")
        if alg == "beamsearch":
            model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold, 'CF' + 'R'*resource + 'O'*outcome)
            beamsearch_cf.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
                                            target_act_to_int, target_int_to_act, target_act_to_int2, target_int_to_act2, res_to_int, target_res_to_int,
                                            target_int_to_res, model_filename, output_filename, bk_filename, method_fitness, resource, outcome, weight)
        # elif alg == "baseline":
        #     model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold, 'CF' + 'R'*resource + 'O'*outcome)
        #     baseline_cfr.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
        #                                     target_act_to_int, target_int_to_act, res_to_int, target_res_to_int,
        #                                     target_int_to_res, model_filename, output_filename) 
 
        else:
            raise RuntimeError(f"No evaluation algorithm called: '{eval_algorithm}'.")

        print("TIME TO FINISH --- %s seconds ---" % (time.time() - start_time))
