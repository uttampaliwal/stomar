"""Operational automation for StoMar (scheduler, daily loop, retraining CLI).

Entry points here are the runtime scripts that keep data, models and paper
state fresh day to day. They are runnable both as modules (``python -m
src.automation.run_daily``) and as files (``python
src/automation/run_daily.py`` — the scripts re-insert the repo root on
``sys.path`` themselves so scheduler/cron launches with arbitrary cwd work).
"""
