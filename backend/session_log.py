"""Append-only event log. One JSON Lines file per session under backend/logs/.

Every upload, detection result, AI proposal and user confirmation is written
here with a UTC timestamp so a session can be reconstructed later.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(os.environ.get("PRISM_LOG_DIR", Path(__file__).parent / "logs"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def log_path(session_id: str) -> Path:
    return LOG_DIR / f"{session_id}.jsonl"


def log_event(session_id: str, event: str, payload: dict) -> str:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = now_iso()
    record = {"timestamp": ts, "session_id": session_id, "event": event, **payload}
    with open(log_path(session_id), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return ts
