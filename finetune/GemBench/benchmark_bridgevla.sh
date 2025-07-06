#!/bin/bash

source ~/.bashrc

# Initialize conda
. ~/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate gembench_bridgevla

export COPPELIASIM_ROOT=/home/ksaha/Research/ModelBasedPlanning/PriorWork/robot-3dlotus/dependencies/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:1.0

seed=200
epoch=40

for split in test_l2 test_l3 test_l4; do
    json_file=assets/taskvars_${split}.json

    taskvars=$(jq -r '.[]' "$json_file")
    for taskvar in $taskvars; do
        python3 client.py \
            --port 13003  \
            --output_file gembench_results/model_${epoch}/seed${seed}/${split}/result.json \
            --microstep_data_dir /home/ksaha/Research/ModelBasedPlanning/PriorWork/robot-3dlotus/data/gembench/test_dataset/microsteps/seed${seed} \
            --taskvar "$taskvar"
    done
done
