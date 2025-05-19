#!/bin/bash

# Grid search script for running multiple experiments
# This script submits multiple jobs with different parameters

# Get the root directory (parent of meta folder)
SCRIPT_DIR=$(dirname $(realpath $0))
ROOT_DIR=$(dirname $(dirname $(realpath $0)))

# Base directory for results
mkdir -p "$ROOT_DIR/grid_search_results"

# Log file
LOG_FILE="$ROOT_DIR/grid_search_results/grid_search_$(date +%Y%m%d_%H%M%S).log"
echo "Starting grid search at $(date)" > $LOG_FILE

# Define parameter grids
MODELS=("BigEfficientVideoModel")
SEQ_LENGTHS=(10 20 40)
LEARNING_RATES=(0.001)
BATCH_SIZES=(16)

# Fixed parameters
SAVE_FORMAT="both"
SAVE_PERIOD=5
NUM_WORKERS=2
EPOCHS=50
WALLTIME="20:00:00"
MEMORY="40gb"
GPUS=2

# Create a timestamp for this grid search
GRID_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Function to submit a job and log it
submit_job() {
    local model=$1
    local seq_length=$2
    local lr=$3
    local batch_size=$4

    # Adjust memory and GPU requirements based on model and sequence length
    local memory=$MEMORY
    local gpus=$GPUS
    local walltime=$WALLTIME

    if [[ "$model" == "BigEfficientVideoModel" && "$seq_length" -ge 50 ]]; then
        memory="40gb"
        gpus=4
        walltime="48:00:00"
    fi

    # Create a unique group name for this configuration
    local group="grid_search_${GRID_TIMESTAMP}"

    echo "Submitting job: model=$model, seq_length=$seq_length, lr=$lr, batch_size=$batch_size" | tee -a $LOG_FILE

    # Update run_config.json with the parameters
    DEFAULT_CONFIG_PATH="${ROOT_DIR}/configs/run_config_default.json"
    CONFIG_PATH="${ROOT_DIR}/configs/run_config_$(date +%Y%m%d_%H%M%S).json"
    python -c "
import json
config = json.load(open('${DEFAULT_CONFIG_PATH}'))
config['model'] = '${model}'
config['seq_length'] = ${seq_length}
config['save_format'] = '${SAVE_FORMAT}'
config['save_period'] = ${SAVE_PERIOD}
config['num_workers'] = ${NUM_WORKERS}
config['lr'] = ${lr}
config['batch_size'] = ${batch_size}
config['epochs'] = ${EPOCHS}
json.dump(config, open('${CONFIG_PATH}', 'w'), indent=4)
"

    #change the permissions
    chmod +x ${SCRIPT_DIR}/submit.sh

    # Submit the job using the submit.sh script
    ${SCRIPT_DIR}/submit.sh --model $model \
                --seq_length $seq_length \
                --walltime $walltime \
                --memory $memory \
                --gpus $gpus \
                --config ${CONFIG_PATH} \
                --group ${group}

    # Get the job ID from the last line of output
    local jobid=$(qstat -u $(whoami) | tail -1 | awk '{print $1}')
    echo "Job submitted with ID: $jobid" | tee -a $LOG_FILE

    # Sleep to avoid overwhelming the scheduler
    sleep 2
}

# Submit all combinations
echo "Submitting grid search jobs..." | tee -a $LOG_FILE

for model in "${MODELS[@]}"; do
    for seq_length in "${SEQ_LENGTHS[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            for batch_size in "${BATCH_SIZES[@]}"; do
                submit_job $model $seq_length $lr $batch_size
            done
        done
    done
done

echo "Grid search submission completed at $(date)" | tee -a $LOG_FILE
echo "Total jobs submitted: $((${#MODELS[@]} * ${#SEQ_LENGTHS[@]} * ${#LEARNING_RATES[@]} * ${#BATCH_SIZES[@]}))" | tee -a $LOG_FILE

# Show current jobs
qstat -u $(whoami) | tee -a $LOG_FILE
