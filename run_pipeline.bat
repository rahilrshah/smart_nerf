@echo off
ECHO =======================================================
ECHO       SMART NeRF - UNIFIED PIPELINE LAUNCHER
ECHO           Enhanced with Parallel Analysis
ECHO =======================================================

REM Check if help was requested
if "%1"=="--help" goto :show_help
if "%1"=="-h" goto :show_help
if "%1"=="/?" goto :show_help

ECHO.
ECHO Activating 'smart_nerf' environment for the complete pipeline...
call conda activate smart_nerf
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate the 'smart_nerf' environment.
    echo Make sure you've run the simple environment update first.
    pause
    exit /b %errorlevel%
)
ECHO 'smart_nerf' environment activated successfully.

ECHO.
ECHO Detecting system specifications for optimal performance...

REM Check if psutil is available for system detection
python -c "import psutil; print(f'System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total/(1024**3):.1f}GB RAM')" 2>nul
if %errorlevel% neq 0 (
    echo Warning: Could not detect system specs. Using default settings.
)

ECHO.
ECHO Running the enhanced pipeline with parallel analysis...
ECHO Arguments passed: %*
ECHO.

python run.py %*

set PY_EXIT_CODE=%errorlevel%
call conda deactivate

if %PY_EXIT_CODE% neq 0 (
    echo ERROR: The pipeline failed.
    echo.
    echo Common solutions:
    echo 1. Check dataset path exists
    echo 2. Ensure all dependencies are installed
    echo 3. Verify GPU drivers are up to date
    echo 4. Try reducing --max_workers or --batch_size for memory issues
    pause
    exit /b %PY_EXIT_CODE%
)

ECHO.
ECHO =======================================================
ECHO   ENHANCED PIPELINE HAS FINISHED SUCCESSFULLY!
ECHO =======================================================
ECHO.
ECHO The pipeline used parallel analysis for faster processing.
ECHO Check the results directory for optimized datasets and meshes.

pause
goto :eof

:show_help
ECHO.
ECHO SMART NeRF Pipeline - Enhanced Version
ECHO.
ECHO Basic Usage:
ECHO   run_pipeline.bat --dataset DATASET_NAME
ECHO.
ECHO Common Examples:
ECHO   run_pipeline.bat --dataset my_object
ECHO   run_pipeline.bat --dataset my_object --mode optimize_only
ECHO   run_pipeline.bat --dataset my_object --max_workers 8 --batch_size 16
ECHO   run_pipeline.bat --dataset my_object --test_views 256 --initial_images 20
ECHO.
ECHO Performance Options:
ECHO   --max_workers N          Number of parallel analysis workers (default: auto-detect)
ECHO   --batch_size N           Viewpoints rendered simultaneously (default: 8)
ECHO   --test_views N           Total viewpoints to analyze (default: 128)
ECHO   --auto_workers           Automatically detect optimal workers (default: true)
ECHO.
ECHO Pipeline Control:
ECHO   --mode MODE              full, optimize_only, or mesh_only (default: full)
ECHO   --initial_images N       Starting dataset size (default: 15)
ECHO   --max_geom_iterations N  Max geometry optimization loops (default: 4)
ECHO   --max_detail_iterations N Max detail refinement loops (default: 4)
ECHO   --max_recommendations N  Max images to add per loop (default: 10)
ECHO.
ECHO Training Control:
ECHO   --geom_training_steps N  Training steps for geometry phase (default: 3500)
ECHO   --detail_training_steps N Training steps for detail phase (default: 5000)
ECHO.
ECHO Mesh Generation:
ECHO   --mesh_method METHOD     Nerfstudio method (default: neus-facto)
ECHO   --mesh_iterations N      Mesh training iterations (default: 20000)
ECHO.
ECHO For more detailed help, see the documentation or run:
ECHO   python run.py --help
ECHO.
pause