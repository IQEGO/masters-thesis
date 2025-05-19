#!/usr/bin/env python3
"""
Grid search script for local training of video assessment models.
Runs multiple training configurations with the same group name for easy comparison.
Works on Windows and Linux with Accelerate.
"""

import argparse
import subprocess
import itertools
import time
import os
import json
import sys
from datetime import datetime

def setup_accelerate_config():
    """Create a non-distributed accelerate config that works on Windows and Linux"""
    config_dir = os.path.expanduser("~/.cache/huggingface/accelerate")
    config_file = os.path.join(config_dir, "default_config.yaml")

    os.makedirs(config_dir, exist_ok=True)

    config_content = """compute_environment: LOCAL_MACHINE
distributed_type: NO
gpu_ids: all
mixed_precision: fp16
num_processes: 1
use_cpu: false
"""

    with open(config_file, 'w') as f:
        f.write(config_content)

    print(f"Created accelerate config at {config_file}")
    return config_file

def run_training(params, group_name):
    """Run a single training job with the given parameters using accelerate"""
    # Ensure accelerate config exists
    setup_accelerate_config()

    # Get the root directory (parent of local folder)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Use raw string for Windows paths to avoid escape character issues
    config_path = os.path.join(root_dir, 'configs', 'full_config.json')

    # Load the full_config.json file
    try:
        with open(config_path, 'r') as f:
            full_config = json.load(f)
    except Exception as e:
        print(f"Error loading full_config.json: {e}")
        full_config = {}

    # Update the full_config with the parameters for this run
    model = params["model"]
    del params["model"]
    full_config["model"] = model
    full_config[model]["model_config"]["dropout"] = params["dropout"]
    full_config[model]["run_config"].update(params)
    full_config['group'] = group_name
    full_config['mode'] = 'local'

    # Save the updated full_config
    with open(config_path, 'w') as f:
        json.dump(full_config, f, indent=4)

    # Build command
    train_script = os.path.join(root_dir, 'common', 'train_loop.py')
    cmd = [
        "accelerate", "launch", train_script
    ]

    # Print the command
    print("\n" + "="*80)
    print(f"Running configuration: {json.dumps(params, indent=2)}")
    print("Command:", " ".join(cmd))
    print("="*80)

    # Run the command
    try:
        start_time = time.time()
        process = subprocess.Popen(cmd)
        process.wait()
        end_time = time.time()

        # Check return code
        if process.returncode != 0:
            print(f"Training failed with return code {process.returncode}")
            return process.returncode

        print(f"Training completed in {(end_time - start_time)/60:.2f} minutes")
        return 0
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Terminating current run...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

        print("Exiting grid search...")
        sys.exit(1)
    except Exception as e:
        print(f"Error running training: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(description="Run grid search for video assessment model training")
    parser.add_argument("--group", type=str, default=f"grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        help="Group name for all runs (default: grid_search_<timestamp>)")
    parser.add_argument("--config", type=str, help="JSON config file with parameter grid")
    parser.add_argument("--no-confirm", action="store_true", help="Skip confirmation prompt and start immediately")
    args = parser.parse_args()

    # Default parameter grid
    param_grid = {
        "model": ["BigEfficientVideoModel"],
        "seq_length": [10],
        "batch_size": [4, 8],
        "epochs": [30],
        "lr": [0.001, 0.0005],
        "weight_decay": [0.01, 0.005],
        "clip_grad_norm": [0.7, 0.5, 0.3],
        "clip_grad_value": [0, 0.1],
        "scheduler": ["reduce_lr_on_plateau", "one_cycle"]
    }

    # Load config file if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                custom_grid = json.load(f)
                # Update the default grid with custom values
                param_grid.update(custom_grid)
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)

    # Generate all combinations of parameters
    keys = param_grid.keys()
    combinations = list(itertools.product(*[param_grid[key] for key in keys]))

    # Create parameter dictionaries for each combination
    all_params = []
    for combo in combinations:
        params = {key: value for key, value in zip(keys, combo)}
        all_params.append(params)

    # Print summary
    print(f"Grid search with group name: {args.group}")
    print(f"Total configurations to run: {len(all_params)}")
    print(f"Estimated total time: {len(all_params) * 30 / 60:.1f} hours (assuming 30 minutes per run)")

    # Ask for confirmation if not using --no-confirm
    if not args.no_confirm:
        response = input("Do you want to proceed? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(0)

    # Run all configurations
    for i, params in enumerate(all_params):
        print(f"\nRunning configuration {i+1}/{len(all_params)}")
        result = run_training(params, args.group)

        # wait 2 seconds to avoid overlapping output
        time.sleep(2)

        # If training was interrupted but user wants to continue, skip to next configuration
        if result != 0:
            print("Continuing with next configuration...")

    print("\nGrid search completed!")

if __name__ == "__main__":
    main()
