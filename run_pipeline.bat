@echo off
ECHO =======================================================
ECHO   Initializing Smart NeRF Pipeline Environment...
ECHO =======================================================

REM This command activates your conda environment.
call conda activate smart_nerf
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to activate the 'smart_nerf' Conda environment.
    pause
    exit /b %errorlevel%
)

REM --- THE CRITICAL NEURALANGELO FIX ---
REM Add the Conda environment's library binaries to the system PATH.
REM This ensures that when our Python script calls Neuralangelo's training script,
REM it can find all the necessary PyTorch and CUDA DLLs.
set PATH=%CONDA_PREFIX%\Library\bin;%PATH%

ECHO.
ECHO Environment activated and PATH configured successfully.
ECHO =======================================================
ECHO   Starting the Python Master Script...
ECHO =======================================================
ECHO.

REM This runs your main Python script, passing along all arguments
REM that you passed to this batch file (e.g., --dataset lego).
python run.py %*

ECHO.
ECHO =======================================================
ECHO   Pipeline has finished.
ECHO =======================================================

REM Pause to keep the window open so you can see the results.
pause