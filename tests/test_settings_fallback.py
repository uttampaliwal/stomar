"""Tests for the pydantic-free ``_FallbackSettings`` path.

The fallback only activates when pydantic/pydantic-settings are unavailable,
so it is exercised in a subprocess where both modules are masked. Bad or
blank STOMAR_* values must fall back to defaults instead of raising at
import time.
"""

import os
import subprocess
import sys
import textwrap

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fallback(env_updates):
    env = os.environ.copy()
    env.update(env_updates)
    env.setdefault("PYTHONPATH", _ROOT)
    code = textwrap.dedent(
        """
        import sys
        sys.modules["pydantic_settings"] = None
        sys.modules["pydantic"] = None
        from src.core.settings import settings
        print(settings.brokerage_rate)
        print(settings.default_epochs)
        print(settings.cache_ttl)
        print(settings.risk_free_rate)
        print(settings.live_poll_interval)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        cwd=_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    return [float(x) for x in proc.stdout.split()]


def test_bad_and_blank_env_vars_fall_back_to_defaults():
    got = _fallback({
        "STOMAR_BROKERAGE_RATE": "0.05%",
        "STOMAR_DEFAULT_EPOCHS": "x",
        "STOMAR_CACHE_TTL": "",
    })
    assert got == [0.0003, 40.0, 30.0, 0.065, 3.0]


def test_valid_env_vars_parse():
    got = _fallback({
        "STOMAR_BROKERAGE_RATE": "0.001",
        "STOMAR_DEFAULT_EPOCHS": "60",
        "STOMAR_CACHE_TTL": "120",
        "STOMAR_RISK_FREE_RATE": "0.07",
        "STOMAR_LIVE_POLL_INTERVAL": "5.0",
    })
    assert got == [0.001, 60.0, 120.0, 0.07, 5.0]


def test_missing_env_vars_use_defaults():
    # point every numeric var at benign values first, then clear them
    env_updates = {
        "STOMAR_BROKERAGE_RATE": "0.01",
        "STOMAR_DEFAULT_EPOCHS": "50",
        "STOMAR_CACHE_TTL": "90",
        "STOMAR_RISK_FREE_RATE": "0.09",
        "STOMAR_LIVE_POLL_INTERVAL": "4.0",
    }
    for var in env_updates:
        env_updates[var] = ""
    got = _fallback(env_updates)
    assert got == [0.0003, 40.0, 30.0, 0.065, 3.0]