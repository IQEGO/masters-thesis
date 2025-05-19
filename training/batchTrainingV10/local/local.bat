@echo off
REM Script to run training locally with Accelerate
REM This will use GPU if available, or fall back to CPU

REM Set default group name if not provided
set GROUP_NAME=%1
if "%GROUP_NAME%"=="" set GROUP_NAME=local_%date:~-4,4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set GROUP_NAME=%GROUP_NAME: =%

echo Starting local training with group name: %GROUP_NAME%

REM Create a simple Accelerate config for non-distributed training
mkdir "%USERPROFILE%\.cache\huggingface\accelerate" 2>nul
echo compute_environment: LOCAL_MACHINE > "%USERPROFILE%\.cache\huggingface\accelerate\default_config.yaml"
echo distributed_type: NO >> "%USERPROFILE%\.cache\huggingface\accelerate\default_config.yaml"
echo gpu_ids: all >> "%USERPROFILE%\.cache\huggingface\accelerate\default_config.yaml"
echo mixed_precision: fp16 >> "%USERPROFILE%\.cache\huggingface\accelerate\default_config.yaml"
echo num_processes: 1 >> "%USERPROFILE%\.cache\huggingface\accelerate\default_config.yaml"
echo use_cpu: false >> "%USERPROFILE%\.cache\huggingface\accelerate\default_config.yaml"

REM Activate venv if it exists
if exist "venv" (
    call venv\Scripts\activate
)

REM Get the root directory (parent of local folder)
set ROOT_DIR=%~dp0..

REM Update run_config.json with the group name
python -c "import json, os; config_path = os.path.join(r'%ROOT_DIR%', 'configs', 'full_config.json'); config = json.load(open(config_path)); config['group'] = '%GROUP_NAME%'; config['mode'] = 'local'; json.dump(config, open(config_path, 'w'), indent=4)"

REM Run the training script with accelerate
accelerate launch %ROOT_DIR%\common\train_loop.py

REM Deactivate venv if it was activated
if exist "venv" (
    call venv\Scripts\deactivate
)

echo Training completed!
