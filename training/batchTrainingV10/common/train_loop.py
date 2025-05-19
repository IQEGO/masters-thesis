import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import importlib
import argparse
import yaml
import json
import time
import signal
import sys
from datetime import datetime
import additional_metrics

# Accelerate imports
from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb

# Add parent directory to path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from common.KonvidVarLenDataset import KonvidVarLenDataset

import torch.multiprocessing as mp
# Only set spawn method if not already set and if we're not on Windows
if not mp.get_start_method(allow_none=True):
    try:
        mp.set_start_method('spawn', force=True)  # Required for num_workers>0 on GPU
    except RuntimeError:
        pass  # Already set by another library


# Global variables for signal handler
_GLOBAL_RESULTS_DIR = None
_GLOBAL_ACCELERATOR = None
_GLOBAL_MODEL = None
_GLOBAL_CONFIG = None
_GLOBAL_RUN_NAME = None
_GLOBAL_MODE = None
_GLOBAL_EPOCH = None

def save_and_kill(signum=None, frame=None):
    """Signal handler to save model when process is killed"""
    # Parameters are required for signal handler interface but not used
    if _GLOBAL_ACCELERATOR is None:
        print("Warning: No accelerator available for saving model")
        sys.exit(0)

    _GLOBAL_ACCELERATOR.print("💥 SIGTERM received - saving model before exiting...")

    if _GLOBAL_ACCELERATOR.is_main_process and wandb.run:
        try:
            # Save final model with both formats
            save_checkpoint(
                _GLOBAL_ACCELERATOR,
                _GLOBAL_MODEL,
                _GLOBAL_EPOCH,
                _GLOBAL_CONFIG,
                _GLOBAL_RUN_NAME,
                _GLOBAL_MODE,
                "both"  # Always save both formats when killed
            )
            _GLOBAL_ACCELERATOR.print("✅ Model saved successfully after SIGTERM")
        except Exception as e:
            _GLOBAL_ACCELERATOR.print(f"❌ Error saving model: {e}")

    sys.exit(0)


def model_fn(model_class_name: str, sequence_length: int, output_dim: int, model_config):
    """Dynamically load and instantiate the model class"""
    # Import from common folder
    module = importlib.import_module(f"common.{model_class_name}")
    ModelClass = getattr(module, model_class_name)

    if model_class_name == "BigEfficientVideoModel":
        return ModelClass(
            encoder_dim=model_config.get("encoder_dim", 512),
            heads=model_config.get("heads", 8),
            depth=model_config.get("depth", 4),
            mlp_dim=model_config.get("mlp_dim", 1024),
            output_dim=output_dim,
            seq_len=sequence_length,
            pos_encoding_type=model_config.get("pos_encoding_type", 'learnable'),
            dropout=model_config.get("dropout", 0.1),
            use_checkpointing=True
        )
    elif model_class_name == "BigFlexibleVideoModel":
        return ModelClass(
            encoder_dim=model_config.get("encoder_dim", 768),
            heads=model_config.get("heads", 4),
            mlp_hidden_dims=model_config.get("mlp_hidden_dims", [256, 64]),
            output_dim=output_dim,
            seq_len=sequence_length,
            pos_encoding_type=model_config.get("pos_encoding_type", 'learnable')
        )
    elif model_class_name == "BigDropoutFlexibleVideoModel":
        return ModelClass(
            encoder_dim=model_config.get("encoder_dim", 768),
            heads=model_config.get("heads", 4),
            mlp_hidden_dims=model_config.get("mlp_hidden_dims", [256, 64]),
            output_dim=output_dim,
            seq_len=sequence_length,
            pos_encoding_type=model_config.get("pos_encoding_type", 'learnable'),
            dropout=model_config.get("dropout", 0.1)
        )
    elif model_class_name == "NormalVideoModel":
        return ModelClass(
            encoder_dim=model_config.get("encoder_dim", 512),
            mlp_dim=model_config.get("mlp_dim", 1024),
            output_dim=output_dim,
            dropout=model_config.get("dropout", 0.2),
            use_checkpointing=True
        )
    else:
        return ModelClass(output_dim = output_dim, sequence_length=sequence_length) #TinyVideoModel

def create_dataloaders(dataset, batch_size, train_split=0.8, val_split=0.1, num_workers=2):
    """
    Create train, validation, and test dataloaders with proper device placement

    Args:
        dataset: The full dataset
        batch_size: Batch size for dataloaders
        train_split: Proportion of data for training (default: 0.8)
        val_split: Proportion of data for validation (default: 0.1)
        num_workers: Number of worker processes for data loading

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Calculate split sizes
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_split)
    val_size = int(dataset_size * val_split)
    test_size = dataset_size - train_size - val_size  # Ensure all samples are used

    # Create the splits with fixed random seed for reproducibility
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Dataset split: {train_size} training, {val_size} validation, {test_size} test samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    # Enable prefetching if supported by the dataset
    if hasattr(dataset, 'prefetch_batch'):
        # Prefetch first few batches for training
        for i in range(min(5, len(train_loader))):
            indices = [train_dataset.indices[j] for j in range(i * batch_size,
                        min((i + 1) * batch_size, len(train_dataset)))]
            dataset.prefetch_batch(indices)

    return train_loader, val_loader, test_loader

def check_distributed_setup(accelerator, model):
    """Check if distributed setup is working correctly"""
    if not accelerator.is_main_process:
        return

    accelerator.print("Checking distributed setup...")
    accelerator.print(f"Distributed setup: {accelerator.distributed_type}")
    accelerator.print(f"Number of processes: {accelerator.num_processes}")
    accelerator.print(f"Process index: {accelerator.process_index}")
    accelerator.print(f"Local process index: {accelerator.local_process_index}")
    accelerator.print(f"Device: {accelerator.device}")

    # Check if model is properly distributed
    if hasattr(accelerator.unwrap_model(model), "device_ids"):
        accelerator.print(f"Model device IDs: {accelerator.unwrap_model(model).device_ids}")

def export_to_onnx(model, filepath, seq_length=10):
    """Export model to ONNX format with error handling"""
    try:
        # Check if onnx is installed
        import onnx
    except ImportError:
        print("ONNX package not installed. Installing now...")
        try:
            import subprocess
            subprocess.check_call(["pip", "install", "onnx"])
            import onnx
            print("ONNX successfully installed")
        except Exception as e:
            print(f"Failed to install ONNX: {e}")
            return None

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Unwrap model if it's wrapped
    if hasattr(model, "module"):
        unwrapped_model = model.module
    else:
        unwrapped_model = model

    # Use model's export method if available
    if hasattr(unwrapped_model, 'export_to_onnx'):
        try:
            unwrapped_model.export_to_onnx(filepath, seq_length)
            return filepath
        except Exception as e:
            print(f"Error in model's export_to_onnx method: {e}")
    try:
        # Fallback to standard ONNX export
        dummy_input = torch.randn(1, seq_length, 3, 224, 224)
        torch.onnx.export(
            unwrapped_model,
            dummy_input,
            filepath,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"Exported model to ONNX format at {filepath}")
        return filepath
    except Exception as e:
        print(f"Error exporting model to ONNX: {e}")
        return None

def save_checkpoint(accelerator, model, epoch, config, run_name, mode="local", save_format="both"):
    """Save a checkpoint using accelerator's built-in state management"""
    if not accelerator.is_main_process:
        return

    # Create checkpoint directory for this run
    checkpoint_dir = os.path.join(_GLOBAL_RESULTS_DIR, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save checkpoint using accelerator
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}")
    os.makedirs(checkpoint_path, exist_ok=True)
    accelerator.save_state(checkpoint_path)
    accelerator.print(f"Saved checkpoint to {checkpoint_path}")

    # Save metadata
    metadata = {
        'epoch': epoch,
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'format': save_format
    }

    with open(os.path.join(checkpoint_dir, f"metadata_epoch{epoch}.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Also save the model in the requested format
    if save_format == "pt" or save_format == "both":
        model_path = os.path.join(checkpoint_dir, f"model_epoch{epoch}.pt")
        accelerator.save(accelerator.unwrap_model(model), model_path)
        accelerator.print(f"Saved PyTorch model to {model_path}")

    # Export to ONNX if requested
    if save_format == "onnx" or save_format == "both":
        onnx_path = os.path.join(checkpoint_dir, f"model_epoch{epoch}.onnx")
        export_to_onnx(
            accelerator.unwrap_model(model),
            onnx_path,
            seq_length=config.get("seq_length", 10)
        )

def load_checkpoint(accelerator, checkpoint_path):
    """Load a checkpoint to resume training"""
    if not os.path.exists(checkpoint_path):
        accelerator.print(f"Checkpoint {checkpoint_path} not found, starting from scratch")
        return 0

    accelerator.print(f"Loading checkpoint from {checkpoint_path}")

    # Load state using accelerator
    accelerator.load_state(checkpoint_path)

    # Try to determine the epoch from the checkpoint path
    try:
        # Extract epoch number from path like "checkpoint_epoch10"
        epoch_str = os.path.basename(checkpoint_path).split('_')[-1].replace('epoch', '')
        epoch = int(epoch_str)
        return epoch + 1  # Return the next epoch to start from
    except (ValueError, IndexError):
        accelerator.print("Could not determine epoch from checkpoint path, starting from epoch 0")
        return 0

def load_config(config_path):
    """Load a configuration file (JSON or YAML)"""
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return {}

    try:
        if config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                return json.load(f)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            print(f"Unsupported config file format: {config_path}")
            return {}
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        return {}

def train_run(config_dict=None):
    """Main training function with distributed training and checkpointing support"""
    global _GLOBAL_RESULTS_DIR, _GLOBAL_ACCELERATOR, _GLOBAL_MODEL, _GLOBAL_CONFIG, _GLOBAL_RUN_NAME, _GLOBAL_MODE, _GLOBAL_EPOCH

    #========================================================================

    # Load configurations from configs folder
    # Use raw strings for Windows paths to avoid escape character issues
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')
    full_config = load_config(os.path.join(config_dir, 'full_config.json'))
    wandb_config = load_config(os.path.join(config_dir, 'wandb_config.json'))
    paths_config = load_config(os.path.join(config_dir, 'paths_config.json'))

    # Override with provided config if any
    if config_dict:
        full_config.update(config_dict)
        
    model_name = full_config.get('model', 'TinyVideoModel')
    mode = full_config.get('mode', 'local')
    group = full_config.get('group', f"default_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    modelwise_config = full_config.get(model_name)
    run_config = modelwise_config.get("run_config")
    model_config = modelwise_config.get("model_config")

    # Extract parameters from run_config ('param_name', 'default_value')
    id_column = run_config.get('id_column', 'flickr_id')
    label_columns = run_config.get('label_columns', ["MOS"])
    seq_length = run_config.get('seq_length', 10)
    save_format = run_config.get('save_format', 'both')
    save_every_n_epochs = run_config.get('save_period', 5)
    num_workers = run_config.get('num_workers', 2)
    resume_from = run_config.get('resume', None)
    scheduler_type = run_config.get("scheduler", "reduce_lr_on_plateau")
    epochs = run_config.get("epochs", 30)
    batch_size = run_config.get("batch_size", 4)
    learning_rate = run_config.get("lr", 0.001)

    # Extract WandB parameters
    entity = wandb_config.get('entity', 'bkubes_masters_thesis')
    project = wandb_config.get('project', 'konvid-test-2')
    api_key = wandb_config.get('api_key', None)

    # Extract paths based on mode
    if mode in paths_config:
        paths = paths_config[mode]
        dataset_path = {}
        # Here create a code to use os.getenv('SCRATCHDIR') a base dir in meta mode, else use absolute path
        if mode == "meta":
            results_dir = os.path.join(os.getenv('SCRATCHDIR'), paths["results"])
            dataset_path["video"] = os.path.join(os.getenv('SCRATCHDIR'), paths["video"])
            dataset_path["csv"] = os.path.join(os.getenv('SCRATCHDIR'), paths["csv"])
        else:
            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), paths["results"])
            # here intelligently detect whether the paths start with ".." and if so, use relative path with os.path.dirname(os.path.abspath(__file__)) as basedir, else use absolute path
            if paths["video"].startswith(".."):
                dataset_path["video"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), paths["video"])
            else:
                dataset_path["video"] = paths["video"]
            if paths["csv"].startswith(".."):
                dataset_path["csv"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), paths["csv"])
            else:
                dataset_path["csv"] = paths["csv"]
    else:
        raise ValueError(f"Invalid mode: {mode}, expected one of {list(paths_config.keys())} (keys in paths_config.json)")

    os.makedirs(results_dir, exist_ok=True)
    _GLOBAL_RESULTS_DIR = results_dir

    #========================================================================

    # Login to WandB if API key is provided
    if api_key:
        wandb.login(key=api_key, relogin=True)

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision="fp16"  # Enable mixed precision for better performance
    )

    # Set global variables for signal handler
    _GLOBAL_ACCELERATOR = accelerator
    _GLOBAL_CONFIG = run_config
    _GLOBAL_MODE = mode

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGTERM, save_and_kill)
    signal.signal(signal.SIGINT, save_and_kill)

    # Set random seed for reproducibility
    set_seed(42)

    # Initialize wandb tracking
    accelerator.init_trackers(
        project,
        init_kwargs={"wandb": {
            "entity": entity,
            "group": group,
            "tags": [model_name.lower()] + label_columns + [str(seq_length)] + ["V10"] + ["additional_metrics"],
            "config": modelwise_config
        }}
    )

    # Get run name from wandb for checkpoint naming
    if accelerator.is_main_process:
        run = accelerator.get_tracker("wandb").run
        run_name = run.name
    else:
        run_name = "run"

    # Update global run name for signal handler
    _GLOBAL_RUN_NAME = run_name

    # Log basic information
    accelerator.print(f"Starting training with model: {model_name}")
    accelerator.print(f"Sequence length: {seq_length}")
    accelerator.print(f"Number of workers: {num_workers}")
    accelerator.print(f"Device count: {torch.cuda.device_count()}")
    accelerator.print(f"Using mixed precision: {accelerator.mixed_precision}")
    accelerator.print(f"Results directory: {results_dir}")
    accelerator.print(f"Run name: {run_name}")

    # Create model, optimizer, and loss function
    model = model_fn(model_name, seq_length, len(label_columns), model_config)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=run_config.get("weight_decay", 1e-2)
    )

    # Loss function
    loss_fn = nn.MSELoss()

    # Create dataset with optimizations
    dataset = KonvidVarLenDataset(
        video_dir=dataset_path["video"],
        labels_csv=dataset_path["csv"],
        id_column=id_column,
        label_columns=label_columns,
        sequence_length=seq_length,
        resize_dim=(224, 224),
        csv_sep=",",
        sampling_strategy='first',
        cache_size=50,  # Cache 50 videos in memory
        enable_prefetch=True,
        num_prefetch_workers=2,
        multiprocessing_context=True if num_workers > 0 else None  # Signal dataset that multiprocessing will be used
    )
    if accelerator.is_main_process:
        try:
            mins, maxs = dataset.get_label_ranges()
            # Log the dictionaries as config parameters (not as metrics)
            run = accelerator.get_tracker("wandb").run
            if run:
                run.config.update({
                    "label_mins": mins,
                    "label_maxs": maxs
                })
        except Exception as e:
            accelerator.print(f"Error logging label ranges: {e}")
            # This is not critical, so we can continue
            pass

    # Create dataloaders (before scheduler so we can calculate total steps for OneCycleLR)
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=batch_size,
        train_split=0.8,
        val_split=0.1,
        num_workers=num_workers
    )

    if scheduler_type == "one_cycle":
        # OneCycleLR scheduler - calls step after each batch
        total_steps = len(train_loader) * epochs
        accelerator.log({
            "train_loader_len": len(train_loader),
            "total_scheduler_steps": total_steps
        })
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=0.3,  # Spend 30% of training increasing LR
            div_factor=25,  # Initial LR is max_lr/25
            final_div_factor=1000,  # Final LR is max_lr/25000
            anneal_strategy='cos'
        )
    else:
        # ReduceLROnPlateau scheduler - calls step after each validation
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=run_config.get("lr_factor", 0.5),
            patience=run_config.get("lr_patience", 5),
            verbose=True
        )

    # Register objects for checkpointing
    accelerator.register_for_checkpointing(scheduler)

    # Prepare for distributed training
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )
    _GLOBAL_MODEL = model

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from:
        checkpoint_path = os.path.join(results_dir, run_name, resume_from)
        if not os.path.exists(checkpoint_path):
            # Try with full path
            checkpoint_path = resume_from

        start_epoch = load_checkpoint(accelerator, checkpoint_path)
        accelerator.print(f"Resuming from epoch {start_epoch}")

    check_distributed_setup(accelerator, model)
    # Training loop
    for epoch in range(start_epoch, epochs):
        _GLOBAL_EPOCH = epoch
        accelerator.print(f"Starting epoch {epoch+1}/{epochs}")
        # Training phase
        model.train()
        train_loss = 0.0

        r2_score = 0.0
        pearson_corr = 0.0
        spearman_corr = 0.0

        start_time = time.time()
        for batch_idx, (x, y) in enumerate(train_loader):
            # Forward pass
            preds = model(x)
            loss = loss_fn(preds, y)

            # Backward pass
            accelerator.backward(loss)

            # Gradient clipping - this should prevent gradients from exploding therefore NaN detection theoretically should not be needed

            # Gradient clipping - L2 Norm (scaling the gradient vector)
            if run_config.get("clip_grad_norm", 0) > 0:
                accelerator.clip_grad_norm_(model.parameters(), run_config.get("clip_grad_norm", 0.7))

            # Gradient clipping - Value (max gradient value for each parameter)
            if run_config.get("clip_grad_value", 0) > 0:
                accelerator.clip_grad_value_(model.parameters(), run_config.get("clip_grad_value", 0))

            # Update weights
            optimizer.step()

            # Step OneCycleLR scheduler if used (needs to be called after each batch)
            if scheduler_type == "one_cycle":
                scheduler.step()

            optimizer.zero_grad()

            # Accumulate loss
            train_loss += loss.item()
            r2_score += additional_metrics.r2_score(y, preds).item()
            pearson_corr += additional_metrics.pearson_corr(y, preds).item()
            spearman_corr += additional_metrics.spearman_corr(y, preds).item()

            # Log batch progress
            if accelerator.is_main_process and batch_idx % 10 == 0:
                accelerator.print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} (batch_size={batch_size})| Loss: {loss.item():.4f}")

        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        avg_train_r2_score = r2_score / len(train_loader)
        avg_train_pearson_corr = pearson_corr / len (train_loader)
        avg_train_spearman_corr = spearman_corr / len(train_loader)
        epoch_time = time.time() - start_time

        # Validation phase
        model.eval()
        val_loss = 0.0
        r2_score = 0.0
        pearson_corr = 0.0
        spearman_corr = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                preds = model(x)
                preds, y = accelerator.gather_for_metrics((preds, y))
                loss = loss_fn(preds, y)
                val_loss += loss.item()
                r2_score += additional_metrics.r2_score(y, preds).item()
                pearson_corr += additional_metrics.pearson_corr(y, preds).item()
                spearman_corr += additional_metrics.spearman_corr(y, preds).item()

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        avg_val_r2_score = r2_score / len(val_loader)
        avg_val_pearson_corr = pearson_corr / len (val_loader)
        avg_val_spearman_corr = spearman_corr / len(val_loader)

        # Update learning rate scheduler (only for ReduceLROnPlateau)
        if scheduler_type != "one_cycle":
            scheduler.step(avg_val_loss)

        # Log metrics
        metrics = {
            "sequence_length": seq_length,
            "initial_learning_rate": learning_rate,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "batch_size": batch_size,
            "epoch": epoch,
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "epoch_time_seconds": epoch_time,
            "samples_per_second": len(train_loader.dataset) / epoch_time,
            "avg_train_r2_score": avg_train_r2_score,
            "avg_train_pearson_corr": avg_train_pearson_corr,
            "avg_train_spearman_corr": avg_train_spearman_corr,
            "avg_val_r2_score": avg_val_r2_score,
            "avg_val_pearson_corr": avg_val_pearson_corr,
            "avg_val_spearman_corr": avg_val_spearman_corr
        }

        accelerator.log(metrics, step=epoch)

        # Save checkpoint if needed
        if save_every_n_epochs > 0 and (epoch + 1) % save_every_n_epochs == 0:
            save_checkpoint(
                accelerator=accelerator,
                model = model,
                epoch=epoch,
                config=run_config,
                run_name=run_name,
                mode=mode,
                save_format=save_format
            )

    # Save final model
    save_checkpoint(
        accelerator=accelerator,
        model = model,
        epoch=epochs,
        config=run_config,
        run_name=run_name,
        mode=mode,
        save_format=save_format
    )

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    r2_score = 0.0
    pearson_corr = 0.0
    spearman_corr = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x)
            preds, y = accelerator.gather_for_metrics((preds, y))
            loss = loss_fn(preds, y)
            test_loss += loss.item()
            r2_score += additional_metrics.r2_score(y, preds).item()
            pearson_corr += additional_metrics.pearson_corr(y, preds).item()
            spearman_corr += additional_metrics.spearman_corr(y, preds).item()

    # Calculate average test loss
    avg_test_loss = test_loss / len(test_loader)
    avg_test_r2_score = r2_score / len(val_loader)
    avg_test_pearson_corr = pearson_corr / len (val_loader)
    avg_test_spearman_corr = spearman_corr / len(val_loader)
    accelerator.log({
        "test_loss": avg_test_loss,
        "avg_test_r2_score": avg_test_r2_score,
        "avg_test_pearson_corr" : avg_test_pearson_corr,
        "avg_test_spearman_corr": avg_test_spearman_corr
    })

    # End training
    accelerator.end_training()

    # Clean up distributed process group
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()

    return accelerator.unwrap_model(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train video quality assessment model")

    # Default config path is in the configs folder
    # Use raw strings for Windows paths to avoid escape character issues
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')
    default_config = os.path.join(config_dir, 'run_config.json')

    parser.add_argument("--config", type=str, default=default_config, help="Path to run configuration file")
    args = parser.parse_args()

    # Load run configuration if provided
    config_dict = None
    config_path = os.path.join(config_dir, args.config)
    if args.config and os.path.exists(config_path):
        config_dict = load_config(config_path)

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGTERM, save_and_kill)

    # Start training
    train_run(config_dict)
