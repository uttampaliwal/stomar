"""Tests for atomic_append_jsonl under concurrent writers.

Multiple processes append large (> PIPE_BUF) JSON lines to the same file;
without the per-file lock these can interleave and corrupt the log.
"""

import json
import multiprocessing

from src.core.secure_io import atomic_append_jsonl

_N_WORKERS = 8
_N_RECORDS = 25
_BIG_FIELD = "x" * 64 * 1024  # 64 KiB payload, well over any atomic chunk


def _worker(path, worker_id):
    for i in range(_N_RECORDS):
        atomic_append_jsonl(path, {"worker": worker_id, "i": i, "blob": _BIG_FIELD})


def test_concurrent_appends_produce_valid_undamaged_lines(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    procs = [
        multiprocessing.Process(target=_worker, args=(path, w))
        for w in range(_N_WORKERS)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    assert all(p.exitcode == 0 for p in procs)

    lines = open(path, encoding="utf-8").read().splitlines()
    assert len(lines) == _N_WORKERS * _N_RECORDS

    # every line must be one complete, parseable record
    records = [json.loads(line) for line in lines]
    assert all(rec["blob"] == _BIG_FIELD for rec in records)

    # all records survived exactly once
    seen = {}
    for rec in records:
        key = (rec["worker"], rec["i"])
        assert key not in seen, f"duplicate record {key}"
        seen[key] = True
    assert len(seen) == _N_WORKERS * _N_RECORDS


def test_single_append_is_fsynced(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    atomic_append_jsonl(path, {"a": 1, "b": "x" * 200_000})
    with open(path, encoding="utf-8") as f:
        assert json.loads(f.read()) == {"a": 1, "b": "x" * 200_000}
