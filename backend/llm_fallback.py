"""AI-assisted column-role proposal (PRISM Step 0 fallback).

Only called when no deterministic signature matches. The model *proposes* a
role per column from a fixed vocabulary; nothing here decides anything. Every
proposal is returned to the user for explicit confirmation or correction.

Parsing is strict: a missing entry, an unknown role, or a malformed answer
becomes "unresolved" -- it is never silently defaulted to a real role.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict, deque

from .format_detect import is_missing_token, looks_numeric
from .models import ROLES, UNRESOLVED, AIProposalBatch

DEFAULT_MODEL = "claude-opus-5"
SAMPLE_ROWS = 15
MAX_CELL_CHARS = 40
COLUMNS_PER_CALL = 120

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
evidence (one short sentence citing the header text and/or the observed \
values that support the role). Return exactly one entry per column, in the \
order given."""


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
    model = os.environ.get("PRISM_LLM_MODEL", DEFAULT_MODEL)
    all_indices = list(range(len(header)))

    if not os.environ.get("ANTHROPIC_API_KEY"):
        why = "AI fallback unavailable (ANTHROPIC_API_KEY is not set). Assign this role manually."
        return {"status": "unavailable", "model": None, "error": why, "calls": [],
                "columns": [_unresolved(i, header[i], why) for i in all_indices]}

    import anthropic

    client = anthropic.Anthropic()
    columns: list[dict] = []
    calls: list[dict] = []
    for start in range(0, len(all_indices), COLUMNS_PER_CALL):
        chunk = all_indices[start:start + COLUMNS_PER_CALL]
        prompt = build_prompt(header, rows, chunk, n_rows_total)
        call_log = {"columns": [header[i] for i in chunk], "prompt": prompt}
        try:
            response = client.messages.parse(
                model=model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_format=AIProposalBatch,
            )
        except anthropic.APIStatusError as e:
            why = f"AI call failed (HTTP {e.status_code}). Assign this role manually."
            call_log["error"] = str(e)
            calls.append(call_log)
            columns += [_unresolved(i, header[i], why) for i in chunk]
            continue
        except anthropic.APIConnectionError as e:
            why = "AI call failed (could not reach the API). Assign this role manually."
            call_log["error"] = str(e)
            calls.append(call_log)
            columns += [_unresolved(i, header[i], why) for i in chunk]
            continue

        call_log["stop_reason"] = response.stop_reason
        call_log["raw_output"] = "".join(b.text for b in response.content if b.type == "text")
        calls.append(call_log)

        if response.stop_reason != "end_turn" or response.parsed_output is None:
            why = f"AI response unusable (stop_reason={response.stop_reason}). Assign this role manually."
            columns += [_unresolved(i, header[i], why) for i in chunk]
            continue
        columns += _reconcile(header, chunk, response.parsed_output)

    failed = sum(1 for c in calls if "error" in c)
    status = "ok" if failed == 0 else ("failed" if failed == len(calls) else "partial")
    error = None if failed == 0 else f"{failed} of {len(calls)} AI call(s) failed; affected columns are unresolved."
    return {"status": status, "model": model, "error": error, "calls": calls, "columns": columns}
