#!/bin/bash
#PBS -j oe

# Detect user and project paths from environment variables
USER=${user:-$(whoami)}
PROJECT_PATH=${project_path:-$(dirname $(dirname $(realpath $0)))}
DATASET_PATH=${dataset_path:-konvidDataset}
SCRIPT_NAME=${script_name:-$(dirname $(dirname $(realpath $0)))/common/train_loop.py}
CONFIG_PATH=${config_path:-$(dirname $(dirname $(realpath $0)))/configs/run_config.json}

# Print job info
echo "🚀 Starting job on $(hostname) at $(date)"
echo "Nvidia smi:"
nvidia-smi

# Set up paths
STORAGE_DIR="/storage/plzen1/home/$USER"
DATASET_DIR="$STORAGE_DIR/$DATASET_PATH"
SCRATCHDIR=${SCRATCHDIR:-/tmp}

echo "📂 Setting up environment in $SCRATCHDIR"
mkdir -p $SCRATCHDIR/codes
mkdir -p $SCRATCHDIR/dataset
mkdir -p $SCRATCHDIR/results

# Copy code files directly (no archive)
echo "🗂️ Copying code files from $PROJECT_PATH"
# Try to use rsync if available (more efficient)
if command -v rsync &> /dev/null; then
    rsync -a --exclude="results" $PROJECT_PATH/ $SCRATCHDIR/codes/
else
    # Fallback to cp + rm if rsync is not available
    cp -r $PROJECT_PATH/* $SCRATCHDIR/codes/
    rm -rf $SCRATCHDIR/codes/results
fi
echo "🗂️ Obsah složky codes:" > $SCRATCHDIR/results/copied_codes.txt
find $SCRATCHDIR/codes/ >> $SCRATCHDIR/results/copied_codes.txt

echo "🗂️ Copying dataset from $DATASET_DIR"
cp -r $DATASET_DIR/* $SCRATCHDIR/dataset/
echo "🗂️ Obsah složky dataset:" > $SCRATCHDIR/results/copied_dataset.txt
find $SCRATCHDIR/dataset/ >> $SCRATCHDIR/results/copied_dataset.txt

# Copy results back to storage if terminated by force after hitting walltime
jobnum=${PBS_JOBID%%.*}
RESULTS_DIR="$PROJECT_PATH/results_$jobnum"
mkdir -p $RESULTS_DIR
cleanup_and_exit() {
    echo "(shell) $(date '+%T')⚠️ SIGTERM zachycen – zálohuji..." >> $SCRATCHDIR/results/log.txt
    pkill -TERM -P $$
    sleep 10
    rsync --remove-source-files -a $SCRATCHDIR/results/* "$RESULTS_DIR/" #rychlejší než cp -r
    clean_scratch
    exit 0
}
trap cleanup_and_exit SIGTERM

SING_IMAGE="/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:24.10-py3.SIF"
IMAGE_BASE=$(basename $SING_IMAGE | tr ':' '_')
ARCHIVE_NAME=".venv_cache_${IMAGE_BASE}.tar.gz"
ARCHIVE_PATH="$STORAGE_DIR/$ARCHIVE_NAME"
PYTHONUSERBASE_HOMEDIR="$STORAGE_DIR/.local-${IMAGE_BASE}"
SCRATCHENV="$SCRATCHDIR/env"
REQUIREMENTS=$PROJECT_PATH/common/requirements.txt

if [ ! -f "$ARCHIVE_PATH" ]; then
    echo "Creating new cached virtualenv at $ARCHIVE_PATH" > $SCRATCHDIR/results/log.txt
    mkdir -p $PYTHONUSERBASE_HOMEDIR
    singularity exec --nv -H $STORAGE_DIR \
        --bind /storage \
        --bind $SCRATCHDIR \
        --env PYTHONUSERBASE=$PYTHONUSERBASE_HOMEDIR \
        $SING_IMAGE pip install --user -r $REQUIREMENTS > $SCRATCHDIR/results/log.txt 2>&1
    if [ -d "$PYTHONUSERBASE_HOMEDIR/lib" ]; then
        tar -czf $ARCHIVE_PATH -C $PYTHONUSERBASE_HOMEDIR .
    else
        echo "❌ Instalace selhala, složka $PYTHONUSERBASE_HOMEDIR je prázdná!" > $SCRATCHDIR/results/log.txt
        exit 1
    fi
fi

mkdir -p $SCRATCHENV
tar -xzf $ARCHIVE_PATH -C $SCRATCHENV

# Detect Python version
PYTHON_VERSION=$(singularity exec $SING_IMAGE python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "➡️  Detected Python version in image: $PYTHON_VERSION" > $SCRATCHDIR/results/log.txt

# Set up environment variables
export PYTHONUSERBASE=$SCRATCHENV
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python${PYTHON_VERSION}/site-packages:$PYTHONPATH
export WANDB_API_KEY=6f8dd0e97364b3cdebfae073827e5d545ceac10c

# Check for CUDA
singularity exec $SING_IMAGE python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')" >> $SCRATCHDIR/results/log.txt

# Change to code directory
cd $SCRATCHDIR/codes

# Configure accelerate for distributed training
echo "⚙️ Configuring accelerate"
cat > accelerate_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: $(nvidia-smi --list-gpus | wc -l)
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

# Run the training script with accelerate
echo "🏃 Running training script with accelerate"
singularity exec --nv -H $SCRATCHDIR \
    --bind /storage \
    --bind $SCRATCHDIR \
    --env PYTHONUSERBASE=$PYTHONUSERBASE \
    --env PATH=$PATH \
    --env PYTHONPATH=$PYTHONPATH \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    $SING_IMAGE accelerate launch \
    --config_file accelerate_config.yaml \
    $SCRIPT_NAME --config $CONFIG_PATH > $SCRATCHDIR/results/log.txt 2>&1

echo "Scratchdir size at the end of the job:"
du -sh $SCRATCHDIR

echo "✅ Job completed at $(date)"

echo "📂 Copying results back to storage"
cp -r $SCRATCHDIR/results/* $RESULTS_DIR/

# Copy wandb logs if they exist
if [ -d "$SCRATCHDIR/codes/wandb" ]; then
    cp -r $SCRATCHDIR/codes/wandb $RESULTS_DIR/
fi

# Copy results from codes if they exist - in case the results are incorrectly saved in codes/results (i.e. in "local" mode)
if [ -d "$SCRATCHDIR/codes/results" ]; then
    echo "Copying from $SCRATCHDIR/codes/results"
    cp -r $SCRATCHDIR/codes/results/* $RESULTS_DIR/
fi

clean_scratch