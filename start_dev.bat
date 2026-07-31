@echo off
setlocal
echo ========================================
echo   StoMar - Full Stack Startup
echo ========================================
echo.

cd /d "%~dp0"

REM --- Ensure uv is installed ---
where uv >nul 2>nul
if errorlevel 1 (
    echo [setup] 'uv' not found. Installing uv (Python package manager)...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo [error] uv installation failed. Install it manually: https://docs.astral.sh/uv/getting-started/installation/
        pause
        exit /b 1
    )
    REM refresh PATH after installer runs
    for /f "delims=" %%i in ('where uv') do set "UV=%%i"
    if not defined UV set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

REM --- Sync Python env (creates .venv with pinned Python, installs deps from uv.lock) ---
echo [setup] Syncing Python dependencies with uv...
uv sync
if errorlevel 1 (
    echo [error] Python dependency sync failed.
    pause
    exit /b 1
)

REM --- Install Node deps if vite missing (npm ci = reproducible from lockfile) ---
if not exist "web\node_modules\.bin\vite.cmd" (
    where node >nul 2>nul
    if errorlevel 1 (
        echo [error] Node.js not found. Install Node 22+ from https://nodejs.org
        pause
        exit /b 1
    )
    echo [setup] Installing Node dependencies...
    pushd web
    call npm ci
    if errorlevel 1 (
        echo [error] Node dependency install failed.
        popd
        pause
        exit /b 1
    )
    popd
)

REM --- Backend: reuse an already-running healthy API, otherwise start one ---
powershell -Command "try { $null = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://localhost:8000/api/health; exit 0 } catch { exit 1 }"
if not errorlevel 1 (
    echo [1/3] API already healthy on :8000 - reusing it
) else (
    echo [1/3] Starting FastAPI backend on :8000...
    start "StoMar API" cmd /k "cd /d %~dp0 && uv run uvicorn api.main:app --reload --port 8000"
    echo [2/3] Waiting for API to start...
    timeout /t 3 /nobreak >nul
)

echo [3/3] Starting React frontend on :5173...
start "StoMar Web" cmd /k "cd /d %~dp0\web && npm run dev"

echo.
echo ========================================
echo   Both servers starting!
echo   API:    http://localhost:8000
echo   UI:     http://localhost:5173
echo   Docs:   http://localhost:8000/docs
echo ========================================
echo.
echo Press any key to close this window (servers keep running).
pause >nul
