"""The run manifest (v3 §6.5): library version, git hash, parameters, seed, the sha256 of every
input file, the overrides hash and timings, so a run can be reproduced and compared."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

from . import __version__
from .util import canonical_json, now_iso, sha256_bytes


def git_hash():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parent.parent),
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5)
        return r.stdout.decode().strip() or None
    except Exception:
        return None


def params_hash(params):
    return sha256_bytes(canonical_json(params).encode("utf-8"))


def build(run_id, session_id, params, inputs, overrides_sha, timings, scope):
    import numpy
    return {"run_id": run_id, "session_id": session_id, "created_at": now_iso(), "prism_version": __version__,
            "git_hash": git_hash(), "python": platform.python_version(), "numpy": numpy.__version__,
            "seed": params.get("seed"), "params": params, "params_sha256": params_hash(params),
            "inputs": inputs, "overrides_sha256": overrides_sha, "scope": scope, "timings_s": timings}
