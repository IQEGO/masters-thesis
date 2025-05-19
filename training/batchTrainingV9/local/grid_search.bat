@echo off
REM Script to run grid search locally with Accelerate on Windows

REM Set group name from command line or use default
set GROUP_NAME=%1
if "%GROUP_NAME%"=="" set GROUP_NAME=grid_search_%date:~-4,4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set GROUP_NAME=%GROUP_NAME: =%

echo Starting grid search with group name: %GROUP_NAME%

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
set CONFIG_DIR=%ROOT_DIR%\configs

REM Check if grid_search_config.json exists in the configs folder
if exist "%CONFIG_DIR%\grid_search_config.json" (
    echo Found grid_search_config.json, using custom configuration
    python grid_search.py --group %GROUP_NAME% --no-confirm --config "%CONFIG_DIR%\grid_search_config.json"
) else (
    echo No config file found, using default parameter grid
    python grid_search.py --group %GROUP_NAME% --no-confirm
)

REM Deactivate venv if it was activated
if exist "venv" (
    call venv\Scripts\deactivate
)

echo Grid search completed!
