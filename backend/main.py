"""PRISM Step 0 -- Data Input & Recognition API.

Run from the repository root:
    uvicorn backend.main:app --reload
then open http://127.0.0.1:8000

Pipeline for one upload:
  1. accept CSV/TSV only; parse every cell as an unmodified string
  2. deterministic signature matching (format_detect.py)
  3. only if nothing matched: AI proposal (llm_fallback.py)
  4. user confirms / corrects every column role -> recognized-structure summary
Every step is logged to backend/logs/<session_id>.jsonl.
"""

from __future__ import annotations

import csv
import io
import os
import uuid
from collections import Counter, OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import format_detect, llm_fallback
from .models import ROLES, UNRESOLVED, ConfirmRequest, StructureSummary, UploadResponse
from .session_log import log_event, log_path, now_iso

MAX_BYTES = int(os.environ.get("PRISM_MAX_UPLOAD_MB", "250")) * 1024 * 1024
PREVIEW_ROWS = 15
MAX_SESSIONS = 20
ACCEPTED_EXTENSIONS = {".csv", ".tsv", ".txt"}
WRONG_FORMAT_MESSAGE = (
    "PRISM accepts quantified tables only, as CSV or TSV (tab-separated .tsv / .txt). "
    "Raw spectra and vendor/binary files (.raw, .d, .wiff, .mzML, .mzXML, .xlsx, ...) are "
    "not processed. Export a quantified protein/peptide/feature table from your analysis "
    "software (e.g. MaxQuant, DIA-NN, Spectronaut, FragPipe, XCMS, MZmine) and upload that."
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="PRISM Step 0 - Data Input & Recognition")

# In-memory session store (a restart clears it; the logs on disk persist).
SESSIONS: "OrderedDict[str, dict]" = OrderedDict()


class InputError(Exception):
    pass


# ---------------------------------------------------------------- parsing

def _pick_csv_delimiter(sample_lines: list[str]) -> str:
    """Choose between comma, semicolon and tab for a .csv file: the candidate
    that appears the same non-zero number of times on the most lines wins."""
    best, best_score = ",", -1
    for d in (",", ";", "\t"):
        counts = [len(next(csv.reader([line], delimiter=d))) - 1 for line in sample_lines if line.strip()]
        if not counts or counts[0] == 0:
            continue
        score = sum(1 for c in counts if c == counts[0]) * 1000 + counts[0]
        if score > best_score:
            best, best_score = d, score
    return best


def parse_table(filename: str, raw: bytes) -> dict:
    ext = Path(filename).suffix.lower()
    if ext not in ACCEPTED_EXTENSIONS:
        raise InputError(f"'{filename}' is not a CSV/TSV file. " + WRONG_FORMAT_MESSAGE)
    if not raw.strip():
        raise InputError("The file is empty.")
    if b"\x00" in raw[:8192]:
        raise InputError(f"'{filename}' looks like a binary file. " + WRONG_FORMAT_MESSAGE)

    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    first_lines = text.splitlines()[:20]
    if ext == ".csv":
        delimiter = _pick_csv_delimiter(first_lines)
    else:
        delimiter = "\t"
        if "\t" not in first_lines[0]:
            raise InputError(
                f"'{filename}' has no tab characters in its header, so it is not a TSV table. "
                + WRONG_FORMAT_MESSAGE)

    # newline="" semantics: let the csv module handle quoted line breaks.
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    try:
        header = next(reader)
    except StopIteration:
        raise InputError("The file has no header row.")
    if len(header) < 2:
        raise InputError(
            "Only one column was found. Check that the file is comma-, semicolon- or tab-separated. "
            + WRONG_FORMAT_MESSAGE)

    rows, blank_lines = [], 0
    for row in reader:
        if not any(cell.strip() for cell in row):
            blank_lines += 1  # skipped for counting only; no data row is altered
            continue
        rows.append(row)

    warnings = []
    ragged = sum(1 for r in rows if len(r) != len(header))
    if ragged:
        warnings.append(f"{ragged} row(s) have a different number of fields than the header ({len(header)}).")
    dupes = [c for c, n in Counter(header).items() if n > 1]
    if dupes:
        warnings.append("Duplicate column names: " + ", ".join(repr(d) for d in dupes[:10]))
    if any(not h.strip() for h in header):
        warnings.append("Some columns have an empty header.")
    if blank_lines:
        warnings.append(f"{blank_lines} blank line(s) were not counted as data rows.")
    if not rows:
        warnings.append("The file has a header but no data rows.")

    return {"header": header, "rows": rows, "delimiter": delimiter, "encoding": encoding, "warnings": warnings}


# ---------------------------------------------------------------- structure

def derive_structure(session: dict, roles: list[str]) -> dict:
    """Describe layout and count samples/features from the confirmed roles.
    Pure counting over the unmodified cells -- nothing is transformed."""
    header, rows = session["header"], session["rows"]
    detection = session.get("detection")

    def col_values(i):
        return [r[i].strip() for r in rows if i < len(r) and r[i].strip()]

    idx_of = lambda role: [i for i, r in enumerate(roles) if r == role]
    value_idx, sample_idx, annot_idx = idx_of("feature_value"), idx_of("sample_id"), idx_of("feature_annotation")

    # Which column identifies a feature?
    feature_id_idx = None
    if detection and detection["feature_id_column"] in header:
        cand = header.index(detection["feature_id_column"])
        if roles[cand] == "feature_annotation":
            feature_id_idx = cand
    if feature_id_idx is None and annot_idx:
        feature_id_idx = annot_idx[0]

    if detection and detection["feature_id_column"] is None and detection["signature"] == "generic_feature_table":
        feature_id = detection["feature_id_note"]
    elif feature_id_idx is not None:
        feature_id = f"'{header[feature_id_idx]}' column"
    else:
        feature_id = "none confirmed (features identified by position)"

    if sample_idx:
        sample_vals = col_values(sample_idx[0])
        n_unique = len(set(sample_vals))
        if n_unique == len(rows):
            layout = "samples_as_rows"
            desc = (f"Wide, one row per sample ('{header[sample_idx[0]]}'); "
                    f"each feature_value column is one feature.")
            samples, features = len(rows), len(value_idx)
            if feature_id_idx is not None and feature_id_idx not in value_idx:
                feature_id = "feature_value column headers"
        else:
            layout = "long"
            desc = (f"Long, several rows per sample ('{header[sample_idx[0]]}' repeats); "
                    "features counted as distinct identifiers.")
            samples = n_unique
            features = len(set(col_values(feature_id_idx))) if feature_id_idx is not None else None
    else:
        layout = "features_as_rows"
        desc = "Wide, one row per feature; each feature_value column is one sample."
        samples, features = len(value_idx), len(rows)

    factors = []
    for i, r in enumerate(roles):
        if r in ("subject_id", "timepoint", "batch", "group_or_outcome"):
            factors.append({"column": header[i], "role": r, "n_levels": len(set(col_values(i)))})

    warnings = []
    if not value_idx:
        warnings.append("No column is labelled feature_value, so there are no measured values.")
    if layout == "features_as_rows" and factors:
        warnings.append(
            "Design columns (subject/timepoint/batch/group) were labelled in a table with one row per "
            "feature; in this layout they describe features, not samples. Sample metadata usually comes "
            "from a separate sample sheet.")
    return {"layout": layout, "layout_description": desc, "sample_count": samples,
            "feature_count": features, "feature_id": feature_id, "design_factors": factors,
            "warnings": warnings}


# ---------------------------------------------------------------- API

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "ai_available": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "ai_model": os.environ.get("PRISM_LLM_MODEL", llm_fallback.DEFAULT_MODEL),
        "roles": list(ROLES),
        "signatures": {k: format_detect.PLATFORM_LABELS[k] for k in format_detect.SIGNATURES},
    }


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    raw = await file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, f"File is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    filename = file.filename or "upload"
    try:
        table = parse_table(filename, raw)
    except InputError as e:
        raise HTTPException(415, str(e))

    session_id = uuid.uuid4().hex[:12]
    header, rows = table["header"], table["rows"]
    log_event(session_id, "upload", {
        "filename": filename, "bytes": len(raw), "delimiter": table["delimiter"],
        "encoding": table["encoding"], "n_rows": len(rows), "n_columns": len(header),
        "header": header, "warnings": table["warnings"],
    })

    session = {"filename": filename, **table, "detection": None, "ai": None}
    warnings = list(table["warnings"])

    # 1) Deterministic detection first. A match is certain: no AI call.
    det = format_detect.detect(header, rows)
    if det is not None:
        method = "signature"
        detection = {
            "signature": det.signature, "platform": det.platform, "omics_type": det.omics_type,
            "feature_id_column": det.feature_id_column, "feature_id_note": det.feature_id_note,
            "n_value_columns": len(det.value_column_indices), "layout": det.layout,
            "also_matched": det.also_matched,
        }
        warnings += det.warnings
        columns = [{"index": r["index"], "column": r["column"], "role": r["role"],
                    "confidence": 1.0, "evidence": r["reason"]} for r in det.roles]
        session["detection"] = detection
        log_event(session_id, "signature_match", {**detection, "roles": columns, "warnings": det.warnings})
        ai = None
    else:
        # 2) AI fallback, proposals only.
        log_event(session_id, "signature_match", {"signature": None, "checked": list(format_detect.SIGNATURES)})
        result = llm_fallback.propose_roles(header, rows, len(rows))
        columns = result["columns"]
        method = "ai" if result["status"] in ("ok", "partial") else "manual"
        ai = {"status": result["status"], "model": result["model"], "error": result["error"],
              "n_unresolved": sum(1 for c in columns if c["role"] == UNRESOLVED)}
        session["ai"] = ai
        log_event(session_id, "ai_proposal", {**ai, "proposals": columns, "calls": result["calls"]})
        if result["status"] == "partial":
            warnings.append(result["error"])

    session.update(method=method, proposal=columns)
    SESSIONS[session_id] = session
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)

    return UploadResponse(
        session_id=session_id, filename=filename, delimiter=table["delimiter"],
        encoding=table["encoding"], n_rows=len(rows), n_columns=len(header), header=header,
        preview_rows=rows[:PREVIEW_ROWS], warnings=warnings, method=method,
        detection=session["detection"], ai=ai, columns=columns, roles_vocabulary=list(ROLES),
    )


@app.post("/api/sessions/{session_id}/confirm", response_model=StructureSummary)
def confirm(session_id: str, body: ConfirmRequest):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "Unknown or expired session. Please upload the file again.")
    header = session["header"]

    by_index = {}
    for c in body.columns:
        if c.index in by_index:
            raise HTTPException(422, f"Column index {c.index} appears more than once.")
        if not (0 <= c.index < len(header)) or header[c.index] != c.column:
            raise HTTPException(422, f"Column {c.index} ('{c.column}') does not match the uploaded file.")
        by_index[c.index] = c.role
    missing = [header[i] for i in range(len(header)) if i not in by_index]
    if missing:
        raise HTTPException(422, "Every column needs a confirmed role. Missing: " + ", ".join(missing[:10]))

    roles = [by_index[i] for i in range(len(header))]
    structure = derive_structure(session, roles)

    decisions = []
    for p, final in zip(session["proposal"], roles):
        if p["role"] == UNRESOLVED:
            decision = "assigned"
        elif p["role"] == final:
            decision = "accepted"
        else:
            decision = "corrected"
        decisions.append({
            "index": p["index"], "column": p["column"],
            "proposed_role": p["role"], "proposed_confidence": p.get("confidence"),
            "proposed_evidence": p.get("evidence"), "final_role": final, "decision": decision,
        })
    changes = sum(1 for d in decisions if d["decision"] != "accepted")

    det = session["detection"]
    if det:
        platform, omics = det["platform"], det["omics_type"]
    else:
        platform, omics = "generic/AI-assisted" if session["method"] == "ai" else "generic/manual", "unspecified"

    confirmed_at = now_iso()
    summary = {
        "session_id": session_id, "confirmed_at": confirmed_at, "filename": session["filename"],
        "method": session["method"], "platform": platform, "omics_type": omics,
        "layout": structure["layout"], "layout_description": structure["layout_description"],
        "sample_count": structure["sample_count"], "feature_count": structure["feature_count"],
        "feature_id": structure["feature_id"], "n_rows": len(session["rows"]), "n_columns": len(header),
        "role_counts": {r: roles.count(r) for r in ROLES if roles.count(r)},
        "design_factors": structure["design_factors"], "warnings": structure["warnings"],
        "columns": [{"index": i, "column": header[i], "role": roles[i]} for i in range(len(header))],
        "changes_from_proposal": changes, "log_file": str(log_path(session_id).name),
    }
    # Log before returning (design principle 4).
    log_event(session_id, "confirmation", {"decisions": decisions, "summary": summary})
    session["confirmed"] = summary
    return StructureSummary(**summary)


@app.get("/api/sessions/{session_id}/log")
def session_log(session_id: str):
    if not session_id.isalnum():
        raise HTTPException(400, "Bad session id.")
    path = log_path(session_id)
    if not path.exists():
        raise HTTPException(404, "No log for this session.")
    import json
    return JSONResponse([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()])


@app.get("/tool")
def tool_page():
    return FileResponse(FRONTEND_DIR / "tool.html")


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
