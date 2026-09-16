@echo off
rem Daily readiness watchdog launcher for Windows Task Scheduler.
rem Portable: resolves the project root from this script's own location,
rem so the same file works in any checkout on any Windows machine.
rem Keeps the schtasks /tr string short (261-char limit) and owns the
rem log redirection here, where cmd.exe interprets it.
set "ROOT=%~dp0.."
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\scripts\readiness_watchdog.py" %* >> "%ROOT%\data\pipeline_logs\readiness_daily.log" 2>&1
