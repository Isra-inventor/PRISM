"""AI-assisted column-role proposal (PRISM Step 0 fallback).

Only called when no deterministic signature matches. The model *proposes* a
role per column from a fixed vocabulary; nothing here decides anything. Every
proposal is returned to the user for explicit confirmation or correction.

Parsing is strict: a missing entry, an unknown role, or a malformed answer
becomes "unresolved" -- it is never silently defaulted to a real role.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

from .format_detect import is_missing_token, looks_numeric
from .models import ROLES, UNRESOLVED, AIProposalBatch

# Tried in order: if a model is overloaded (503), rate-limited (429) or retired
# (404) PRISM moves on to the next one. Override with PRISM_LLM_MODEL, e.g.
# PRISM_LLM_MODEL=gemini-3.6-flash or a comma-separated list.
DEFAULT_MODELS = ",".join([
    "gemini-flash-lite-latest",   # lite models: no thinking, answer in seconds
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",           # full flash models: often slow/overloaded
    "gemini-3-flash-preview",
    "gemini-3.8-flash",
])
DEFAULT_MODEL = DEFAULT_MODELS.split(",")[0]
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
REQUEST_TIMEOUT_S = 60        # a slower model is abandoned for the next one
RETRY_STATUS = {429, 500, 502, 503, 504}   # transient: rate limit / overload
RETRY_DELAYS_S = (2,)            # one quick retry per model, then the next model
MAX_RATE_LIMIT_WAITS = 3         # on 429, wait as long as Google asks (<= 35s), up to 3 times
MAX_RATE_LIMIT_WAIT_S = 35
NEXT_MODEL_STATUS = {404, 429, 500, 502, 503, 504}

log = logging.getLogger("prism.ai")


def say(msg):
    """Print an AI-fallback status line to the server terminal (always shown,
    independent of any logging configuration)."""
    print(f"[PRISM AI] {msg}", file=sys.stderr, flush=True)
SAMPLE_ROWS = 15
MAX_CELL_CHARS = 40
COLUMNS_PER_CALL = 100       # fewer requests: free keys allow ~15/min per model
PARALLEL_CALLS = 4          # wide tables: several column batches at once

SYSTEM_PROMPT = f"""You label the columns of a quantified omics data table \
(proteomics or metabolomics) that a scientist has uploaded. You do not \
transform, clean, or judge the data; you only propose what role each column \
plays.

Allowed roles (use exactly these strings):
- sample_id: identifies the biological/analytical sample (one per injection / \
aliquot).
- subject_id: identifies the individual (patient, animal, donor) a sample came \
from; may repeat across samples (e.g. repeated visits).
- timepoint: visit, day, time, or ordered sampling occasion.
- batch: processing / acquisition batch, plate, run order group, instrument day.
- group_or_outcome: experimental group, condition, treatment, phenotype, \
clinical outcome, or other study variable of interest.
- feature_value: a measured quantity (intensity, abundance, area, \
concentration). In a wide table either every sample column or every \
feature column is a feature_value.
- feature_annotation: describes the feature (protein ID, gene, m/z, RT, \
compound name, description, flags).
- ignore: clearly irrelevant to analysis (row index, empty column, export \
artefact).
- {UNRESOLVED}: use this whenever you cannot confidently assign one of the \
roles above. Do not guess. An honest "{UNRESOLVED}" is always preferred over \
a plausible-looking wrong role.

For each column return: column (the exact column name as given), \
proposed_role, confidence (0-1, your probability the role is correct), and \
evidence (at most 15 words citing the header text and/or the observed \
values that support the role). Return exactly one entry per column, in the \
order given."""


# Gemini structured-output schema (OpenAPI subset). One object per column.
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "columns": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "column": {"type": "STRING"},
                    "proposed_role": {"type": "STRING", "enum": list(ROLES) + [UNRESOLVED]},
                    "confidence": {"type": "NUMBER"},
                    "evidence": {"type": "STRING"},
                },
                "required": ["column", "proposed_role", "confidence", "evidence"],
                "propertyOrdering": ["column", "proposed_role", "confidence", "evidence"],
            },
        }
    },
    "required": ["columns"],
}


def api_key():
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _api_error(err):
    """(message, retry_after_seconds_or_None) from a Gemini HTTPError body."""
    body = err.read().decode("utf-8", "replace")
    try:
        e = json.loads(body)["error"]
    except (ValueError, KeyError, TypeError):
        return body[:300] or str(err), None
    delay = None
    for d in e.get("details") or []:
        if str(d.get("@type", "")).endswith("RetryInfo"):
            try:
                delay = float(str(d.get("retryDelay", "")).rstrip("s"))
            except ValueError:
                pass
    return e.get("message", body[:300]), delay


def _api_message(err):
    return _api_error(err)[0]


class GeminiError(RuntimeError):
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


def model_list():
    raw = os.environ.get("PRISM_LLM_MODEL") or DEFAULT_MODELS
    return [m.strip() for m in raw.split(",") if m.strip()]


class ModelOrder:
    """Shared, thread-safe model order: the last model that worked is tried
    first by later batches, so a big table doesn't re-probe busy models."""

    def __init__(self, models):
        self._models = list(models)
        self._lock = threading.Lock()

    def current(self):
        with self._lock:
            return list(self._models)

    def promote(self, model):
        with self._lock:
            if model in self._models:
                self._models.remove(model)
                self._models.insert(0, model)


def call_with_fallback(models, key, system, prompt, order=None):
    """Try each model in turn. Returns (raw_text, finish_reason, model_used)."""
    if order is not None:
        models = order.current()
    last = None
    for model in models:
        try:
            raw, finish = call_gemini_with_retry(model, key, system, prompt)
            if order is not None:
                order.promote(model)
            return raw, finish, model
        except GeminiError as e:
            last = e
            if e.code not in NEXT_MODEL_STATUS or model == models[-1]:
                raise
            say(f"{e} -> switching to next model")
    raise last


def call_gemini_with_retry(model, key, system, prompt):
    """call_gemini, retrying transient errors (429/5xx, network) with backoff.
    Raises RuntimeError with a readable message once retries are exhausted."""
    rate_waits = 0
    attempt = 0
    while True:
        say(f"calling {model} ...")
        try:
            result = call_gemini(model, key, system, prompt)
            say(f"{model} answered (finish_reason={result[1]})")
            return result
        except urllib.error.HTTPError as e:
            code = e.code
            message, retry_after = _api_error(e)
            msg = f"Gemini API error (HTTP {code}, model {model}): {message.splitlines()[0]}"
            if code == 429 and retry_after is not None and rate_waits < MAX_RATE_LIMIT_WAITS \
                    and retry_after <= MAX_RATE_LIMIT_WAIT_S:
                rate_waits += 1
                say(f"{model}: free-tier rate limit reached; Google asks to wait {retry_after:.0f}s -> waiting")
                time.sleep(retry_after + 1)
                continue
            retryable = code in RETRY_STATUS
        except (urllib.error.URLError, OSError) as e:
            reason = getattr(e, "reason", e)
            if isinstance(e, TimeoutError) or "timed out" in str(reason):
                code = 504  # too slow right now: move straight on to the next model
                msg = f"Gemini model {model} did not answer within {REQUEST_TIMEOUT_S}s"
                raise GeminiError(msg, code)
            code = None
            msg = f"Could not reach the Gemini API ({type(e).__name__}): {reason}"
            retryable = True
        if not retryable or attempt >= len(RETRY_DELAYS_S):
            raise GeminiError(msg, code)
        delay = RETRY_DELAYS_S[attempt]
        attempt += 1
        say(f"{msg} -> retrying in {delay}s")
        time.sleep(delay)


def call_gemini(model, key, system, prompt):
    """One generateContent call over plain HTTPS (stdlib only, Python 3.7+).
    Returns (raw_text, finish_reason). Raises urllib.error.HTTPError / URLError."""
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
            "temperature": 0,
            "maxOutputTokens": 32768,   # a full batch of proposals must fit
        },
    }
    req = urllib.request.Request(
        GEMINI_URL.format(model=model),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    candidates = data.get("candidates") or []
    if not candidates:
        reason = (data.get("promptFeedback") or {}).get("blockReason", "NO_CANDIDATES")
        return "", reason
    cand = candidates[0]
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    return text, cand.get("finishReason", "UNKNOWN")


def _column_profile(idx: int, rows: list[list[str]]) -> dict:
    cells = [r[idx] if idx < len(r) else "" for r in rows]
    present = [c for c in cells if not is_missing_token(c)]
    numeric = sum(looks_numeric(c) for c in present)
    return {
        "n_inspected": len(cells),
        "n_missing_like": len(cells) - len(present),
        "n_unique": len(set(present)),
        "fraction_numeric": round(numeric / len(present), 2) if present else None,
    }


def build_prompt(header: list[str], rows: list[list[str]], indices: list[int], n_rows_total: int) -> str:
    sample = rows[:SAMPLE_ROWS]
    profile_rows = rows[:500]
    lines = [
        f"The table has {n_rows_total} data rows and {len(header)} columns in total. "
        f"This request covers {len(indices)} of those columns.",
        "",
        "Columns (name, profile over the first rows, then sample values from the first "
        f"{len(sample)} rows; long values are cut at {MAX_CELL_CHARS} characters):",
    ]
    for i in indices:
        values = [(r[i] if i < len(r) else "")[:MAX_CELL_CHARS] for r in sample]
        lines.append(json.dumps({
            "column": header[i],
            "position": i,
            "profile": _column_profile(i, profile_rows),
            "sample_values": values,
        }, ensure_ascii=False))
    return "\n".join(lines)


def _unresolved(index: int, column: str, why: str) -> dict:
    return {"index": index, "column": column, "role": UNRESOLVED, "confidence": 0.0, "evidence": why}


def _reconcile(header: list[str], indices: list[int], batch: AIProposalBatch) -> list[dict]:
    """Map model output back onto columns. Anything off-spec becomes unresolved."""
    by_name: dict[str, deque] = defaultdict(deque)
    for p in batch.columns:
        by_name[p.column].append(p)

    out = []
    for i in indices:
        col = header[i]
        if not by_name[col]:
            out.append(_unresolved(i, col, "The model returned no proposal for this column."))
            continue
        p = by_name[col].popleft()
        p.confidence = min(max(p.confidence, 0.0), 1.0)
        if p.proposed_role == UNRESOLVED:
            out.append({"index": i, "column": col, "role": UNRESOLVED,
                        "confidence": p.confidence, "evidence": p.evidence})
        elif p.proposed_role not in ROLES:
            out.append(_unresolved(
                i, col, f"The model answered '{p.proposed_role}', which is not an allowed role."))
        else:
            out.append({"index": i, "column": col, "role": p.proposed_role,
                        "confidence": p.confidence, "evidence": p.evidence})
    return out


def propose_roles(header: list[str], rows: list[list[str]], n_rows_total: int) -> dict:
    """Ask the LLM for column-role proposals.

    Returns {"status", "model", "columns", "calls", "error"}. On any failure
    every column comes back unresolved so the user assigns roles manually.
    """
    models = model_list()
    used = []
    all_indices = list(range(len(header)))

    key = api_key()
    if not key:
        why = "AI fallback unavailable (GEMINI_API_KEY is not set). Assign this role manually."
        say("FAILED: GEMINI_API_KEY is not set. Put GEMINI_API_KEY=... in the .env file next to README.md "
            "and restart the server.")
        return {"status": "unavailable", "model": None, "error": why, "calls": [],
                "columns": [_unresolved(i, header[i], why) for i in all_indices]}

    order = ModelOrder(models)
    chunks = [all_indices[i:i + COLUMNS_PER_CALL] for i in range(0, len(all_indices), COLUMNS_PER_CALL)]
    say(f"asking Gemini about {len(all_indices)} column(s) in {len(chunks)} batch(es); "
        f"models to try: {', '.join(models)}")

    def run_chunk(n, chunk):
        """Returns (call_log, column proposals) for one batch of columns."""
        tag = f"batch {n}/{len(chunks)}"
        prompt = build_prompt(header, rows, chunk, n_rows_total)
        call_log = {"columns": [header[i] for i in chunk], "prompt": prompt}
        try:
            raw, finish, model_used = call_with_fallback(models, key, SYSTEM_PROMPT, prompt, order)
        except Exception as e:  # anything at all: report it, never crash the upload
            msg = str(e) if isinstance(e, GeminiError) else f"Unexpected error: {type(e).__name__}: {e}"
            say(f"{tag} FAILED: {msg}")
            if not isinstance(e, GeminiError):
                traceback.print_exc()
            call_log["error"] = msg
            return call_log, [_unresolved(i, header[i], f"{msg} -- assign this role manually.") for i in chunk]

        call_log.update(model=model_used, finish_reason=finish, raw_output=raw)
        parsed = None
        if finish == "STOP":
            try:
                parsed = AIProposalBatch.model_validate_json(raw)
            except ValueError:
                parsed = None
        if parsed is None:
            say(f"{tag} FAILED: unusable response from {model_used} (finish_reason={finish}). "
                f"Raw text: {raw[:300]!r}")
            call_log["error"] = f"unusable response (finish_reason={finish})"
            why = f"AI response unusable (finish_reason={finish}). Assign this role manually."
            return call_log, [_unresolved(i, header[i], why) for i in chunk]
        say(f"{tag} OK: {len(parsed.columns)} proposal(s) from {model_used}")
        return call_log, _reconcile(header, chunk, parsed)

    with ThreadPoolExecutor(max_workers=PARALLEL_CALLS) as pool:
        results = list(pool.map(lambda nc: run_chunk(*nc), enumerate(chunks, 1)))

    calls, columns, used = [], [], []
    for call_log, cols in results:
        calls.append(call_log)
        columns += cols
        m = call_log.get("model")
        if m and m not in used:
            used.append(m)

    failed = sum(1 for c in calls if "error" in c)
    status = "ok" if failed == 0 else ("failed" if failed == len(calls) else "partial")
    first_error = next((c["error"] for c in calls if "error" in c), None)
    error = None if failed == 0 else (
        f"{failed} of {len(calls)} AI call(s) failed; affected columns are unresolved. {first_error}")
    model = ", ".join(used) if used else models[0]
    return {"status": status, "model": model, "error": error, "calls": calls, "columns": columns}
