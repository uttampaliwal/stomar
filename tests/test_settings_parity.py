"""Parity: pydantic ``Settings`` vs the pydantic-free ``_FallbackSettings``.

The hand-written fallback in ``src/core/settings.py`` exists so imports don't
crash in pydantic-free environments (e.g. minimal cron/CLI contexts). Its
defaults must never drift from the pydantic ``Settings`` class, which is the
single source of truth. Each new ``STOMAR_*`` knob added to ``Settings`` must
also be added to ``_FallbackSettings`` — this test makes forgetting it a red
CI build instead of a silent default mismatch at runtime.

Uses the same subprocess masking technique as ``test_settings_fallback.py``
to load the module with ``pydantic``/``pydantic_settings`` unavailable.
"""

import json
import os
import subprocess
import sys
import textwrap

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pydantic_defaults():
    from src.core.settings import Settings

    fields = {}
    for name, info in Settings.model_fields.items():
        if info.is_required():
            fields[name] = None  # no required fields exist today; keep visible
        else:
            fields[name] = info.default
    return fields


def _fallback_defaults():
    env = {k: v for k, v in os.environ.items() if not k.startswith("STOMAR_")}
    env["PYTHONPATH"] = _ROOT
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(
            """
            import json, sys
            sys.modules["pydantic_settings"] = None
            sys.modules["pydantic"] = None
            from src.core.settings import settings
            s = settings.__dict__
            print(json.dumps(
                {k: (v if isinstance(v, (str, int, float, bool, list)) else str(v))
                 for k, v in s.items()}
            ))
            """
        )],
        env=env,
        capture_output=True,
        text=True,
        cwd=_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _equal(pydantic_value, fallback_value):
    if isinstance(pydantic_value, (int, float)) and not isinstance(pydantic_value, bool) \
            and isinstance(fallback_value, (int, float)) and not isinstance(fallback_value, bool):
        # pydantic may keep an int literal where the annotation is float
        return float(pydantic_value) == float(fallback_value)
    return pydantic_value == fallback_value


def test_fallback_field_names_match_pydantic_settings():
    pyd = _pydantic_defaults()
    fb = _fallback_defaults()
    missing = sorted(set(pyd) - set(fb))
    extra = sorted(set(fb) - set(pyd))
    assert not missing, f"Fields present in Settings but missing from _FallbackSettings: {missing}"
    assert not extra, f"Fields present in _FallbackSettings but missing from Settings: {extra}"


def test_fallback_defaults_match_pydantic_settings():
    pyd = _pydantic_defaults()
    fb = _fallback_defaults()
    common = set(pyd) & set(fb)
    drifted = sorted(n for n in common if not _equal(pyd[n], fb[n]))
    assert not drifted, (
        "Default drift between Settings and _FallbackSettings — update the "
        "fallback in src/core/settings.py: "
        + "; ".join(f"{n}: pydantic={pyd[n]!r} fallback={fb[n]!r}" for n in drifted)
    )
