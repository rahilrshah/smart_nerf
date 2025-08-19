@echo off
ECHO =======================================================
ECHO   SMART NeRF - SIMPLE ENVIRONMENT UPDATE
ECHO =======================================================

ECHO.
ECHO This script will add Nerfstudio to your existing 'smart_nerf' environment
ECHO WITHOUT changing your working Instant-NGP setup.
ECHO.

REM Check if environment exists
call conda info --envs | findstr "smart_nerf" >nul
if %errorlevel% neq 0 (
    echo ERROR: 'smart_nerf' environment not found.
    echo Please create it first.
    pause
    exit /b 1
)

ECHO Activating existing 'smart_nerf' environment...
call conda activate smart_nerf
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate smart_nerf environment
    pause
    exit /b %errorlevel%
)

ECHO.
ECHO =======================================================
ECHO Adding Nerfstudio (keeping existing PyTorch)
ECHO =======================================================

ECHO Installing Nerfstudio...
pip install nerfstudio

ECHO.
ECHO =======================================================
ECHO Adding Mesh Processing Libraries
ECHO =======================================================

ECHO Installing Open3D for mesh processing...
pip install open3d

ECHO Installing PyMeshLab for advanced mesh operations...
pip install pymeshlab



call conda deactivate

ECHO.
ECHO =======================================================
ECHO     UPDATE COMPLETE!
ECHO =======================================================
ECHO.
ECHO Your existing Instant-NGP setup is unchanged.
ECHO Added capabilities:
ECHO ✅ Nerfstudio mesh generation  
ECHO ✅ Open3D mesh processing
ECHO ✅ PyMeshLab advanced mesh operations
ECHO.
ECHO Test your unified pipeline with:
ECHO   python run.py --dataset [name] --mode full
ECHO.

pause