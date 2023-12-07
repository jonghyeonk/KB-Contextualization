import time
from pathlib import Path

from src.commons.log_utils import LogData
from src.commons import shared_variables as shared
from src.commons.utils import extract_petrinet_filename , extract_tree_filename , extract_last_model_checkpoint
from src.evaluation.prepare_data import prepare_testing_data, select_petrinet_compliant_traces
from src.evaluation.inference_algorithms import beamsearch_cf_fitness, beamsearch_cf_token, baseline_cf, baseline_cfr, baseline_cfrt5, beamsearch_cf, beamsearch_cfr, baseline_cfrt, \
    baseline_cfrt2, baseline_cfrt3, baseline_cfrt4, baseline_cfrt5, beamsearch_cfrt, beamsearch_cfrt2 ,beamsearch_cfrt3, beamsearch_cfrt4, beamsearch_cfrt5


def evaluate_all(log_data: LogData, models_folder: str, tree: bool):
    start_time = time.time()

    training_traces = log_data.log[log_data.log[log_data.case_name_key].isin(log_data.training_trace_ids)]
    # +1 accounts of fake '!' symbol added to identify end of trace
    maxlen = max([trace.shape[0] for _, trace in training_traces.groupby(log_data.case_name_key)]) + 1
    predict_size = maxlen
    

    act_to_int, target_act_to_int, target_int_to_act, target_act_to_int2, target_int_to_act2 = prepare_testing_data(log_data, training_traces)
    check_new_act = log_data.log[log_data.act_name_key].unique().tolist()
    new_chars = [na for na in check_new_act if na not in target_act_to_int]
    log_data.new_chars = new_chars
    
    
    pn_filename = extract_petrinet_filename(log_data.log_name.value)

    if tree == False:
        pt_filename = None
    else: 
        pt_filename = extract_tree_filename(log_data.log_name.value)
    # Extract evaluation sequences compliant to the background knowledge
    evaluation_traces = log_data.log[log_data.log[log_data.case_name_key].isin(log_data.evaluation_trace_ids)]
    compliant_traces = evaluation_traces

    # jh_eval = ["case_103_test", "case_104_test", "case_111_test", "case_115_test", "case_127_test"]
    # evaluation_traces = log_data.log[log_data.log[log_data.case_name_key].isin(jh_eval)]
    # compliant_traces = evaluation_traces
    
    print("Compliant traces: " + str(compliant_traces[log_data.case_name_key].nunique())
          + " out of " + str(len(log_data.evaluation_trace_ids)))
    print('Elapsed time:', time.time() - start_time)

    for fold in range(shared.folds):
        # for eval_algorithm in [baseline_cf, baseline_cfr, beamsearch_cf, beamsearch_cfr]:
        for eval_algorithm in [beamsearch_cf_fitness]: #beamsearch_cf_fitness
                            #    beamsearch_cfrt3, beamsearch_cfrt4, beamsearch_cfrt5]:
        # for eval_algorithm in [beamsearch_cfrt]:
            start_time = time.time()

            algorithm_name = Path(eval_algorithm.__file__).stem
            folder_path = shared.output_folder / models_folder / str(fold) / 'results' / algorithm_name
            if not Path.exists(folder_path):
                Path.mkdir(folder_path, parents=True)
            output_filename = folder_path / f'{log_data.log_name.value}_beam{str(shared.beam_size)}_fold{str(fold)}.csv'

            print(f"fold {fold} - {algorithm_name}")
            if eval_algorithm in [beamsearch_cf, beamsearch_cf_fitness, beamsearch_cf_token]:
                model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold, 'CF')
                eval_algorithm.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
                                               target_act_to_int2, target_int_to_act2, model_filename, output_filename, pn_filename, pt_filename)
            elif eval_algorithm is beamsearch_cfr:
                model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold, 'CFR')
                beamsearch_cfr.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
                                               target_act_to_int, target_int_to_act, res_to_int, target_res_to_int,
                                               target_int_to_res, model_filename, output_filename, pn_filename, pt_filename)
            elif eval_algorithm is beamsearch_cfrt:
                model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold, 'CFRT')
                beamsearch_cfrt.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
                                               target_act_to_int, target_int_to_act, res_to_int, target_res_to_int,
                                               target_int_to_res, model_filename, output_filename, pn_filename, pt_filename)
            elif eval_algorithm is beamsearch_cfrt2:
                model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold, 'CFRT2')
                beamsearch_cfrt2.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
                                               target_act_to_int, target_int_to_act, res_to_int, target_res_to_int,
                                               target_int_to_res, model_filename, output_filename, pn_filename, pt_filename)
            elif eval_algorithm is beamsearch_cfrt3:
                model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold, 'CFRT3')
                beamsearch_cfrt3.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
                                               target_act_to_int, target_int_to_act, res_to_int, target_res_to_int,
                                               target_int_to_res, model_filename, output_filename, pn_filename, pt_filename)
            elif eval_algorithm is beamsearch_cfrt4:
                model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold, 'CFRT4')
                beamsearch_cfrt4.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
                                               target_act_to_int, target_int_to_act, res_to_int, target_res_to_int,
                                               target_int_to_res, model_filename, output_filename, pn_filename, pt_filename)
            elif eval_algorithm is beamsearch_cfrt5:
                model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold, 'CFRT5')
                beamsearch_cfrt5.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
                                               target_act_to_int, target_int_to_act, res_to_int, target_res_to_int,
                                               target_int_to_res, model_filename, output_filename, pn_filename, pt_filename)    
            elif eval_algorithm is baseline_cf:
                model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold, 'CF')
                baseline_cf.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
                                             target_act_to_int, target_int_to_act, model_filename, output_filename) 
            elif eval_algorithm is baseline_cfr:
                model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold, 'CFR')
                baseline_cfr.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
                                             target_act_to_int, target_int_to_act, res_to_int, target_res_to_int,
                                             target_int_to_res, model_filename, output_filename) 
            elif eval_algorithm is baseline_cfrt:
                model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold, 'CFRT')
                baseline_cfrt.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
                                             target_act_to_int, target_int_to_act, res_to_int, target_res_to_int,
                                             target_int_to_res, model_filename, output_filename)   
            elif eval_algorithm is baseline_cfrt2:
                model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold, 'CFRT2')
                baseline_cfrt2.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
                                             target_act_to_int, target_int_to_act, res_to_int, target_res_to_int,
                                             target_int_to_res, model_filename, output_filename)   
            elif eval_algorithm is baseline_cfrt3:
                model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold, 'CFRT3')
                baseline_cfrt3.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
                                             target_act_to_int, target_int_to_act, res_to_int, target_res_to_int,
                                             target_int_to_res, model_filename, output_filename)   
            elif eval_algorithm is baseline_cfrt4:
                model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold, 'CFRT4')
                baseline_cfrt4.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
                                             target_act_to_int, target_int_to_act, res_to_int, target_res_to_int,
                                             target_int_to_res, model_filename, output_filename)   
            elif eval_algorithm is baseline_cfrt5:
                model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold, 'CFRT5')
                baseline_cfrt5.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
                                             target_act_to_int, target_int_to_act, res_to_int, target_res_to_int,
                                             target_int_to_res, model_filename, output_filename)   
            else:
                raise RuntimeError(f"No evaluation algorithm called: '{eval_algorithm}'.")

            print("TIME TO FINISH --- %s seconds ---" % (time.time() - start_time))
