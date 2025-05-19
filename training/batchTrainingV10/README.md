# KoNViD Video Quality Assessment Framework

A comprehensive framework for training and evaluating video quality assessment models on the KoNViD-1k dataset, with support for both local and Metacentrum environments.

## Project Structure

```
batchTrainingV9/
├── configs/               # Configuration files
│   ├── run_config.json    # Training parameters for local training local.bat
│   ├── wandb_config.json  # Weights & Biases settings
│   ├── paths_config.json  # Dataset paths
│   └── grid_search_config.json  # Grid search parameters for local grid_search.bat
│
├── common/                # Shared code
│   ├── train_loop.py      # Main training script
│   ├── BigEfficientVideoModel.py  # Transformer-based model
│   ├── NormalVideoModel.py        # CNN-based model without transformers
│   ├── KonvidVarLenDataset.py     # Dataset loader
│   └── ...                # Other model implementations
│
├── local/                 # Scripts for local training
│   ├── local.bat          # Run single training job
│   ├── grid_search.bat    # Run grid search
│   └── grid_search.py     # Grid search implementation
│
└── meta/                  # Scripts for Metacentrum
    ├── meta.sh            # Main Metacentrum script
    ├── submit.sh          # Job submission script
    └── grid_search.sh     # Grid search for Metacentrum
```

## Available Models

- **BigEfficientVideoModel**: MobileNetV2 + Transformer architecture
- **NormalVideoModel**: MobileNetV2 + CNN for temporal processing (no transformers)
- **PretrainedVisionTransformer**: Uses a pretrained ViT model - currently not working, under development
- **TinyVideoModel**: Lightweight model for quick experiments

## Configuration Files

### run_config.json

Contains all training parameters:

```json
{
    "model": "BigEfficientVideoModel",  // Model architecture to use
    "seq_length": 10,                   // Number of frames to process
    "save_format": "both",              // Save format (pt, onnx, or both)
    "save_period": 5,                   // Save checkpoint every N epochs
    "num_workers": 2,                   // DataLoader workers
    "mode": "local",                    // Training mode (local or meta)
    "lr": 0.001,                        // Learning rate
    "batch_size": 4,                    // Batch size
    "epochs": 30,                       // Number of epochs
    "scheduler": "reduce_lr_on_plateau", // LR scheduler type
    "weight_decay": 0.0001,             // Weight decay for regularization
    "clip_grad_norm": 0.7,              // Gradient norm clipping (0 to disable)
    "clip_grad_value": 0                // Gradient value clipping (0 to disable)
}
```

Note: clip_grad_norm and clip_grad_value can't be used together, one of them has to be set to 0.

### wandb_config.json

Weights & Biases configuration:

```json
{
    "api_key": "your_wandb_api_key_here",
    "entity": "your_wandb_entity",
    "project": "your_wandb_project"
}
```

### paths_config.json

Dataset and results paths for different environments:

```json
{
    "local": {
        "__comment": "For local mode, use absolute paths, or relative paths starting with '..' ",
        "video": "../../KoNViD_1k_videos",
        "csv": "../../KoNViD_1k_metadata/KoNViD_1k_attributes.csv",
        "results": "../results"
    },
    "meta": {
        "__comment": "For meta mode, use relative paths, the $SCRATCHDIR will be used as base directory",
        "video": "dataset/KoNViD_1k_videos",
        "csv": "dataset/KoNViD_1k_metadata/KoNViD_1k_attributes.csv",
        "results": "results"
    }
}
```

## Usage Guide

### Local Training

1. **Single Training Run**:
   ```
   cd local
   local.bat [optional_group_name]
   ```
   
   This will:
   - Create an Accelerate configuration for distributed training
   - Update run_config.json with the group name
   - Run the training script with the specified configuration

2. **Grid Search**:
   ```
   cd local
   grid_search.bat [optional_group_name]
   ```
   
   This will:
   - Run multiple training configurations sequentially
   - Use grid_search_config.json if it exists
   - Group all runs under the same group name in W&B

#### How It Works

These scripts create a non-distributed Accelerate configuration that works on Windows and properly uses GPU when available. The configuration is stored at:

```
%USERPROFILE%\.cache\huggingface\accelerate\default_config.yaml
```

Key settings:
- `distributed_type: NO` - Avoids the libuv error on Windows
- `gpu_ids: all` - Uses all available GPUs
- `mixed_precision: fp16` - Enables mixed precision for better performance
- `use_cpu: false` - Prefers GPU over CPU when available

#### Stopping a Run

To stop a running training job or grid search:
- Press Ctrl+C in the terminal
- For grid search, you'll need to press Ctrl+C twice to fully exit

### Metacentrum Training

1. **Single Job Submission**:
   ```
   cd meta
   chmod +x submit.sh
   ./submit.sh --model TinyVideoModel --seq_length 10 --walltime "3:00:00" [...and other optional parameters]
   ```
   
   Parameters:
   - `--model TinyVideoModel` - Model architecture (default: BigEfficientVideoModel)
   - `--seq_length 10` - Number of frames to process (default: 10)
   - `--walltime "48:00:00"` - Job time limit (default: 48 hours)
   - `--memory "30gb"` - Memory allocation (default: 30GB)
   - `--scratch_memory "30gb"` - Scratch memory allocation
   - `--gpus 2` - Number of GPUs (default: 2)
   - `--cpus 2` - Number of CPUs (default: 2)
   - `--config` - Path to custom configuration file (optional)
   - `--group` - Group name for W&B (optional)

2. **Grid Search on Metacentrum**:
   ```
   cd meta
   chmod +x grid_search.sh
   ./grid_search.sh
   ```
   
   This will:
   - Submit multiple jobs with different configurations

### Advanced Usage

#### Resuming Training
UNDER DEVELOPMENT

To resume from a checkpoint (implemented with Accelerate checkpointing):

1. Update run_config.json:
   ```json
   {
       "resume": "results/run_name/checkpoint_epoch10"
   }
   ```

2. Run training as usual:
   ```
   cd local
   local.bat
   ```

#### Monitoring Training

Training progress is logged to Weights & Biases. You can view:
- Loss curves
- Learning rate changes
- GPU memory usage

## Model Export

Models are automatically saved in both PyTorch (.pt) and ONNX formats if `save_format` is set to "both". ONNX models can be used for deployment in various environments.

## Requirements

- PyTorch, depending on Metacentrum Singularity image version
- Accelerate
- Weights & Biases
- torchvision
- decord
- onnx

For Metacentrum, all dependencies are installed automatically in chosen Singularity container.
