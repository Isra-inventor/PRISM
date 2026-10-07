"""Append-only JSON Lines records: the audit ledger (every tool call) and session logs."""

from __future__ import annotations

import json
from pathlib import Path

from .util import canonical_json, now_iso, sha256_bytes


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class Ledger:
    """ledger.jsonl of an audit run: tool, parameters, who asked (user / ai), timestamp, output hash."""

    def __init__(self, path):
        self.path = Path(path)

    def record(self, tool, params, who="user", output=None):
        rec = {"tool": tool, "params": params, "who": who, "at": now_iso(),
               "output_sha256": sha256_bytes(canonical_json(output).encode("utf-8")) if output is not None else None}
        append_jsonl(self.path, rec)
        return rec

    def read(self):
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]
