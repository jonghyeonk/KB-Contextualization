#!/usr/bin/env bash

log_names=(
        "5x5_1W"
        "5x5_2W"
        "5x5_1S"
        "5x5_2S"
        "10x2_1S"
        "10x2_1W"
        "10x2_2S"
        "10x2_2W"
        "10x5_1W"
        "10x5_2W"
        "10x5_1S"
        "10x5_2S"
        "10x20_1W"
        "10x20_2W"
        "10x20_1S"
        "10x20_2S"
        "50x5_1W"
        "50x5_2W"
        "50x5_1S"
        "50x5_2S"
        "BPI2012_1W"
        "BPI2012_1S"
        #"BPI2012_1W_bis"
        "BPI2017_1W"
        "BPI2017_1S"
        )

for log_name in "${log_names[@]}"
do
    python experiments_runner.py --log="${log_name}" --use_old_model --full_run &
done