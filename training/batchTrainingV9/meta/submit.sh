#!/bin/bash

# Default values
MODEL="BigEfficientVideoModel"
SEQ_LENGTH=10
WALLTIME="48:00:00"
MEMORY="40gb"
SCRATCH_MEMORY="40gb"
GPUS=2
CPUS=2

# Create timestamp for group name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GROUP="${MODEL}_${SEQ_LENGTH}_${TIMESTAMP}"

# Get the root directory (parent of meta folder)
ROOT_DIR=$(dirname $(dirname $(realpath $0)))
CONFIG_PATH="${ROOT_DIR}/configs/run_config_default.json"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model)
            MODEL="$2"
            shift
            shift
            ;;
        --seq_length)
            SEQ_LENGTH="$2"
            shift
            shift
            ;;
        --walltime)
            WALLTIME="$2"
            shift
            shift
            ;;
        --memory)
            MEMORY="$2"
            shift
            shift
            ;;
        --scratch_memory)
            SCRATCH_MEMORY="$2"
            shift
            shift
            ;;
        --gpus)
            GPUS="$2"
            shift
            shift
            ;;
        --cpus)
            CPUS="$2"
            shift
            shift
            ;;
        --config)
            CONFIG_PATH="$2"
            shift
            shift
            ;;
        --group)
            GROUP="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Determine GPU capability based on model
GPU_CAP="compute_70"
if [[ "$MODEL" == "BigEfficientVideoModel" && "$SEQ_LENGTH" -ge 50 ]]; then
    # Request higher memory GPUs for large models
    GPU_CAP="compute_80"
    MEMORY="60gb"
    SCRATCH_MEMORY="60gb"
    GPUS=4
    CPUS=4
fi

# Put group name into run_config.json
python -c "import json; config = json.load(open('${CONFIG_PATH}')); config['group'] = '${GROUP}'; config['mode'] = 'meta'; json.dump(config, open('${CONFIG_PATH}', 'w'), indent=4)"

# Submit the job
echo "Submitting job with the following parameters:"
echo "Walltime: $WALLTIME"
echo "Memory: $MEMORY"
echo "Scratch memory: $SCRATCH_MEMORY"
echo "GPUs: $GPUS"
echo "CPUs: $CPUS"

PROJECT_DIR="$(dirname $(dirname $(realpath $0)))" # Get the project directory (dir above)

# Create the qsub command
QSUB_CMD="qsub -N PyTorch_Training_${MODEL}_${SEQ_LENGTH} \
     -q gpu \
     -l select=1:ncpus=${CPUS}:ngpus=${GPUS}:mem=${MEMORY}:scratch_local=${SCRATCH_MEMORY}:gpu_cap=${GPU_CAP}:singularity=True \
     -l walltime=${WALLTIME} \
     -M bkubes@students.zcu.cz -m bae \
     -v project_path=${PROJECT_DIR},dataset_path=konvidDataset,script_name=common/train_loop.py,config_path=${CONFIG_PATH}"

# Add the script path
QSUB_CMD="${QSUB_CMD} ${ROOT_DIR}/meta/meta.sh"

# Submit the job
jobid=$(eval "$QSUB_CMD")
jobnum=${jobid%%.*}
echo "Job submitted with ID: $jobid"

# Create results directory
RESULTS_DIR="${PROJECT_DIR}/results_$jobnum"
mkdir -p "$RESULTS_DIR"
qalter -o "$RESULTS_DIR/o_$jobnum.log" \
     -e "$RESULTS_DIR/e_$jobnum.log" $jobid

# Show current jobs
qstat -u $(whoami)
