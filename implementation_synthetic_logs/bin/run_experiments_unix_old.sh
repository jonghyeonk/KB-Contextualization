#!/usr/bin/env bash

log_names=(
  "Data-flow log.xes"
)

for log_name in "${log_names[@]}"
do
    python experiments_runner.py --log="${log_name}" --full_run --use_old_model
done

# python result_parser.py --logs 10x2_1S,10x2_1W,10x2_2S,10x2_2W --target_model old_model --reference_model zeros --table_caption "Old model, AntonDecl, Ivan Checker." --table_label tb:old_model
