@echo off
echo ========================================
echo   StoMar - Full Stack Startup
echo ========================================
echo.

echo [1/3] Starting FastAPI backend on :8000...
start "StoMar API" cmd /c "cd /d %~dp0 && python -m uvicorn api.main:app --reload --port 8000"

echo [2/3] Waiting for API to start...
timeout /t 3 /nobreak >nul

echo [3/3] Starting React frontend on :5173...
start "StoMar Web" cmd /c "cd /d %~dp0\web && npm run dev"

echo.
echo ========================================
echo   Both servers starting!
echo   API:    http://localhost:8000
echo   UI:     http://localhost:5173
echo   Docs:   http://localhost:8000/docs
echo ========================================
echo.
echo Press any key to exit...
pause >nul
