@echo off
REM Optional GPU acceleration for StoMar (RTX 50-series / CUDA 12.8+ machines).
REM Detects an NVIDIA GPU and swaps the CPU-only torch for a CUDA build.
REM Run once per GPU machine:  scripts\setup_gpu.bat
REM Safe to skip on CPU-only machines — StoMar works fine without it.
setlocal

where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo No NVIDIA GPU detected - CPU-only torch stays (nothing to do).
    exit /b 0
)

echo NVIDIA GPU detected:
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

set VENV_PY=.venv\Scripts\python.exe
if not exist "%VENV_PY%" (
    echo Error: .venv not found. Run: uv sync  (or pip install -e .^)
    exit /b 1
)

echo Swapping torch -^> CUDA 12.8 build (cu128, supports RTX 50-series^)...
"%VENV_PY%" -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 "torch>=2.0,<3"

echo.
echo Verifying...
"%VENV_PY%" -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

"%VENV_PY%" -c "import torch; exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo WARNING: CUDA still unavailable - check driver version (needs ^>= 570 for RTX 50-series^).
) else (
    echo GPU acceleration enabled. The auto-pipeline will use CUDA automatically.
)

endlocal
