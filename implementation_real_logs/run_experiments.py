import argparse
import tensorflow as tf
import statistics as stat

from src.commons import log_utils, shared_variables as shared
from src.evaluation import evaluation
from src.training import train_model


class ExperimentRunner:
    def __init__(self, model, port, python_port, train, evaluate):
        self._model = model
        self._port = port
        self._python_port = python_port
        self._train = train
        self._evaluate = evaluate

        print(args.port, python_port)
        print('Used network:', self._model)
        print('Perform training:', self._train)
        print('Perform evaluation:', self._evaluate)

    def run_experiments(self, log_list, alg, method_fitness, weight, resource, timestamp, outcome):
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4,
                                          allow_soft_placement=True)
        session = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(session)

        for log_name in log_list:
            log_path = shared.log_folder / log_name
            self._run_single_experiment(log_path, alg, method_fitness, weight, resource, timestamp, outcome)

    def _run_single_experiment(self, log_path, alg, method_fitness, weight, resource, timestamp, outcome):
        log_data = log_utils.LogData(log_path)
        log_data.encode_log(resource, timestamp, outcome)
        
        trace_sizes = list(log_data.log.value_counts(subset=[log_data.case_name_key], sort=False))

        print('Log name:', log_data.log_name.value + log_data.log_ext.value)
        print('Log size:', len(trace_sizes))
        print('Trace size avg.:', stat.mean(trace_sizes))
        print('Trace size stddev.:', stat.stdev(trace_sizes))
        print('Trace size min.:', min(trace_sizes))
        print('Trace size max.:', max(trace_sizes))
        print(f'Evaluation prefix range: [{log_data.evaluation_prefix_start}, {log_data.evaluation_prefix_end}]')
    
        if self._train:
            train_model.train(log_data, self._model, resource, outcome)
        
        if self._evaluate:
            evaluation.evaluate_all(log_data, self._model, alg, method_fitness, weight, resource, timestamp, outcome)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default=None, help='input log')
    parser.add_argument('--model', default="keras_trans", help='choose among ["LSTM", "custom_trans", "keras_trans"]')
    parser.add_argument('--port', type=int, default=25333, help='communication port (python port = port + 1)')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', default=False, action='store_true', help='train without evaluating')
    group.add_argument('--evaluate', default=True, action='store_true', help='evaluate without training')
    group.add_argument('--full_run', default=False, action='store_true', help='train and evaluate model')

    args = parser.parse_args()

    logs = [args.log.strip()] if args.log else shared.log_list
    
    if args.full_run:
        args.train = True
        args.evaluate = True

    ExperimentRunner(model=args.model,
                     port=args.port,
                     python_port=args.port+1,
                     train=args.train,
                     evaluate=args.evaluate) \
        .run_experiments(log_list=logs,
                         alg = "beamsearch",
                         method_fitness = "fitness_token_based_replay",  
                         weight = [0, 0.95] ,
                         resource = False,
                         timestamp = False,
                         outcome = False)
        
        ## Note
        # alg = "baseline" or "beamsearch"
        # method_fitness = "conformance_diagnostics_alignments", "conformance_diagnostics_token_based_replay", "fitness_alignments", "fitness_token_based_replay"
