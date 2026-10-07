"""Small shared helpers: hashing, canonical JSON, atomic writes, timestamps."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

TIMESTAMP_KEYS = {"at", "answered_at", "imported_at", "created_at", "confirmed_at", "updated_at", "timestamp"}


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def strip_keys(obj, keys):
    if isinstance(obj, dict):
        return {k: strip_keys(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [strip_keys(v, keys) for v in obj]
    return obj


def schema_sha256(schema):
    """Fingerprint of a schema: canonical JSON without timestamps and without the 'ai' block."""
    s = {k: v for k, v in schema.items() if k != "ai"}
    return sha256_bytes(canonical_json(strip_keys(s, TIMESTAMP_KEYS)).encode("utf-8"))


def write_json(path, obj, indent=2):
    write_bytes(path, (json.dumps(obj, indent=indent, ensure_ascii=False) + "\n").encode("utf-8"))


def write_bytes(path, data):
    """Atomic write: a reader never sees a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, str(path))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
