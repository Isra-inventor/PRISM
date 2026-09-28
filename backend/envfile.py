""".env loading that copes with Windows editors (BOM, UTF-16, hidden .txt)."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
KEY_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER")
DOTENV_REPORT = []  # what the .env loader saw; printed at startup if no key is found


def _read_text_any_encoding(path):
    """.env files made on Windows may be UTF-8 with a BOM (Notepad) or UTF-16
    (PowerShell '>' redirection). Handle all of them."""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    if len(raw) > 1 and raw[1:2] == b"\x00":           # UTF-16-LE without BOM
        return raw.decode("utf-16-le")
    try:
        return raw.decode("utf-8-sig")                  # strips a UTF-8 BOM
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _load_dotenv(path=PROJECT_DIR / ".env"):
    """Read KEY=VALUE lines from the project's .env (if any) into the environment.
    A non-empty variable already set in the environment wins. .env is git-ignored."""
    DOTENV_REPORT.append(f"looked for {path} -> {'found' if path.exists() else 'NOT FOUND'}")
    if not path.exists():
        # Windows hides extensions, so Notepad's ".env.txt" looks like ".env": accept it.
        alt = path.with_name(".env.txt")
        if not alt.exists():
            return
        DOTENV_REPORT.append(f"using {alt.name} instead")
        path = alt
    names = []
    for line in _read_text_any_encoding(path).splitlines():
        line = line.strip().lstrip("\ufeff")
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k.lower().startswith("export "):
            k = k[len("export "):].strip()
        if k.lower().startswith("$env:"):              # PowerShell-style line pasted into .env
            k = k[len("$env:"):]
        v = v.strip().strip('"').strip("'").strip()
        names.append(f"{k}{'' if v else ' (EMPTY)'}")
        if v and not os.environ.get(k):
            os.environ[k] = v
    DOTENV_REPORT.append("variables in .env: " + (", ".join(names) if names else "none (file is empty?)"))
    if not any(k.upper() in KEY_NAMES for k in (n.split(" ")[0] for n in names)):
        DOTENV_REPORT.append("no line starting with GEMINI_API_KEY= (check the spelling)")


