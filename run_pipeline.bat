@echo off
ECHO =======================================================
ECHO   Initializing Smart NeRF Pipeline Environment...
ECHO =======================================================

REM This command activates your conda environment and sets up all necessary paths.
REM The "call" is essential - it ensures the script continues after activation.
call conda activate smart_nerf

REM Check if activation was successful
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to activate the 'smart_nerf' Conda environment.
    echo Please make sure you have created the environment correctly.
    pause
    exit /b %errorlevel%
)

ECHO.
ECHO Environment activated successfully.
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