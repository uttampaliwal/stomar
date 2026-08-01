"""Secure file I/O helpers: SHA-256 verification and atomic writes.

Used by the model artifact layer and state persistence paths so that:

* no file is ever deserialized before its bytes match a trusted digest, and
* no state file is ever visible half-written (atomic tmp + os.replace).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from typing import Any, Callable

logger = logging.getLogger(__name__)

_DEFAULT_ATOMIC_MODE = 0o644


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Return the hex SHA-256 digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_sha256(path: str, expected: str) -> bool:
    """Return True only if the file's digest matches *expected* (hex)."""
    if not expected:
        return False
    try:
        return sha256_file(path) == expected.lower()
    except OSError:
        return False


def atomic_write_text(path: str, content: str, mode: int = _DEFAULT_ATOMIC_MODE) -> None:
    """Write *content* to *path* atomically (tmp file + os.replace + fsync)."""
    dir_name = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_json(path: str, data: Any, mode: int = _DEFAULT_ATOMIC_MODE) -> None:
    """Write *data* as pretty JSON to *path* atomically."""
    atomic_write_text(path, json.dumps(data, indent=2, default=str), mode=mode)


def atomic_append_jsonl(path: str, record: dict) -> None:
    """Append a single JSON line to an audit-style JSONL file with fsync.

    Appends are intentionally not os.replace-based (that would clobber the
    file on concurrent appends); the O_APPEND + fsync sequence guarantees
    the line is durable once this returns.
    """
    dir_name = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(dir_name, exist_ok=True)
    line = json.dumps(record, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def atomic_write_bytes(path: str, data: bytes, mode: int = _DEFAULT_ATOMIC_MODE) -> None:
    """Write *data* to *path* atomically (binary)."""
    dir_name = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def safe_path_join(root: str, *parts: str) -> str:
    """Join paths and refuse any traversal outside *root*.

    Raises ValueError when the resolved path would escape the root
    directory (used to keep model artifact access confined).
    """
    root_abs = os.path.abspath(root)
    joined = os.path.abspath(os.path.join(root_abs, *parts))
    if joined != root_abs and not joined.startswith(root_abs + os.sep):
        raise ValueError(f"path {joined!r} escapes allowed root {root_abs!r}")
    return joined


def with_retry(func: Callable[[], Any], attempts: int = 3, exc: type[BaseException] = OSError):
    """Retry a file operation a few times (helps on NFS/btrfs hiccups)."""
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return func()
        except exc as e:  # noqa: PERF203
            last = e
            logger.debug("file op failed (attempt %d/%d): %s", attempt + 1, attempts, e)
    if last is not None:
        raise last
    raise RuntimeError("unreachable")
