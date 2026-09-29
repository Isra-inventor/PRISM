"""AI layer (spec 4): label pre-built column groups from a statistics digest.

What is sent: file-level layout hints and, per group, its name pattern, a few
column names and aggregated statistics. Never raw rows. Example values only
for low-cardinality text columns (<= 20 distinct values), and only when
AI_SEND_EXAMPLE_VALUES is on (default).

The prompt is a briefing (briefing.md + the closed sets + the current scope)
rather than enumerated categories: descriptive fields are free text.
What comes back is parsed strictly (Pydantic) and checked structurally
(validation.py): unknown group ids are rejected, missing groups and values
outside the closed sets become 'unresolved'.
"""

from __future__ import annotations

import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ValidationError

from . import llm_providers as llm
from .config import SCOPE_DESCRIPTION
from .schema import DEFINITIONS, PROMPT_VERSION, UNRESOLVED, VOCABULARY
from .validation import timepoint_detail, validate_feature_identity, validate_group

CACHE_DIR = Path(os.environ.get("PRISM_CACHE_DIR", Path(__file__).parent / "cache"))
GROUPS_PER_CALL = 120
SAMPLE_NAMES_IN_DIGEST = 300


def send_examples():
    return os.environ.get("AI_SEND_EXAMPLE_VALUES", "true").strip().lower() not in ("0", "false", "no", "off")


# ---------------------------------------------------------------- digest

def _group_digest(g, examples):
    d = {
        "group_id": g["group_id"],
        "kind": g["kind"],
        "type": g["type"],
        "n_columns": g["n_columns"],
        "pattern": (g.get("pattern") or {}).get("text"),
        "pattern_side": (g.get("pattern") or {}).get("side"),
        "origin": g["origin"],
    }
    cols = g["columns"]
    if len(cols) <= 5 or (not g.get("pattern") and len(cols) <= 500):
        # no shared name pattern: the names are the only semantic signal (headers, not data)
        d["columns"] = cols
    else:
        d["first_columns"] = cols[:3]
        d["last_columns"] = cols[-2:]
    prof = dict(g.get("profile") or {})
    prof.pop("column", None)
    prof.pop("position", None)
    if not examples or (prof.get("unique_ratio") or 0) >= 0.5:
        # identifier-like columns (e.g. patient codes) never send example values
        prof.pop("values", None)
    if not examples:
        prof.pop("value_shapes", None)
    d["profile"] = prof
    if g.get("sample_names"):
        d["sample_names_after_stripping_pattern"] = g["sample_names"][:SAMPLE_NAMES_IN_DIGEST]
    if g.get("split_from"):
        d["split_from"] = g["split_from"]
        d["split_reasons"] = g.get("split_reasons")
    return d


def build_digest(filename, groups, hints, fixed=None, group_ids=None, signature_hint=None):
    examples = send_examples()
    chosen = [g for g in groups if group_ids is None or g["group_id"] in group_ids]
    return {
        "file": {"name_extension": Path(filename).suffix.lower(), "layout_hints": hints,
                 "signature_hint": signature_hint},
        "already_confirmed": fixed or {},
        "groups": [_group_digest(g, examples) for g in chosen],
        "settings": {"example_values_sent": examples, "raw_rows_sent": False},
    }


def digest_hash(digest):
    return hashlib.sha256(json.dumps(digest, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- prompt

BRIEFING = (Path(__file__).parent / "briefing.md").read_text(encoding="utf-8")


def system_prompt():
    closed = VOCABULARY
    return f"""{BRIEFING}

## Current scope of PRISM's later steps
{SCOPE_DESCRIPTION}
Anything else must still be recognized and described, but flagged in_supported_scope "no" (or
"unsure") with a short scope_reason.

## Closed fields (use exactly these values; code branches on them)
- layout: {closed['layout']}
- role: {closed['column_role']}
- block_role (role value only): {closed['block_role']}. At most one primary block per assay.
- audit_kind (role sample_metadata only, else null): {closed['audit_kind']} ('other' is the escape hatch)
- in_supported_scope: {closed['in_supported_scope']}
- marks_rows_as_suspect (feature_annotation only): true when the column flags rows as decoy /
  contaminant / otherwise suspect.
- is_study_sample: false for QC, blank, pool, calibrator and similar injections.

## Open fields (free text, your own words)
assay_label, omics_type, source_software (add "(unconfirmed)" when inferred from names),
scope_reason, and every group label. A value block's label should describe the measurement and
what the statistics suggest, e.g. "LFQ intensity, apparently raw linear scale (median 2.1e7,
18% zeros)". A time point's label should say what kind of time it is, e.g. "visit label T1/T2
(ordinal)".

## Output
- layout, with confidence and evidence.
- assays: one entry per assay (a file can hold several, e.g. proteins and metabolites side by
  side). feature_identity lists the group_ids whose values identify each feature (several =
  composite key such as m/z + retention time); use [] when feature names are column headers.
- groups: exactly one entry for EVERY group in the digest. Value groups name their assay_label.
  suggest_split may list column names inside a multi-column group that do not belong (e.g. a
  clinical covariate among metabolites), with suggest_split_role / suggest_split_audit_kind /
  suggest_split_label for them.
- samples: sample names or glob patterns (e.g. "QC_*") with a label and is_study_sample.
- clarifying_questions for anything you cannot resolve from the digest.
- Respect everything under already_confirmed."""


def response_schema():
    s = lambda **k: dict(type="STRING", **k)
    n = lambda: {"type": "NUMBER"}
    b = lambda **k: dict(type="BOOLEAN", **k)
    fact = {"type": "OBJECT", "properties": {"value": s(enum=VOCABULARY["layout"]), "confidence": n(),
                                             "evidence": s()}, "required": ["value", "confidence", "evidence"]}
    assay = {"type": "OBJECT", "properties": {
        "assay_label": s(), "omics_type": s(), "source_software": s(),
        "in_supported_scope": s(enum=VOCABULARY["in_supported_scope"]), "scope_reason": s(),
        "feature_identity": {"type": "OBJECT", "properties": {
            "group_ids": {"type": "ARRAY", "items": s()}, "composite": b()}, "required": ["group_ids"]},
        "confidence": n(), "evidence": s()},
        "required": ["assay_label", "omics_type", "in_supported_scope", "confidence", "evidence"]}
    group = {"type": "OBJECT", "properties": {
        "group_id": s(),
        "role": s(enum=VOCABULARY["column_role"]),
        "assay_label": s(nullable=True),
        "label": s(),
        "block_role": s(enum=VOCABULARY["block_role"], nullable=True),
        "audit_kind": s(enum=VOCABULARY["audit_kind"], nullable=True),
        "marks_rows_as_suspect": b(nullable=True),
        "confidence": n(),
        "evidence": s(),
        "suggest_split": {"type": "ARRAY", "items": s(), "nullable": True},
        "suggest_split_role": s(enum=VOCABULARY["column_role"], nullable=True),
        "suggest_split_audit_kind": s(enum=VOCABULARY["audit_kind"], nullable=True),
        "suggest_split_label": s(nullable=True),
    }, "required": ["group_id", "role", "label", "confidence", "evidence"]}
    return {"type": "OBJECT", "properties": {
        "layout": fact,
        "assays": {"type": "ARRAY", "items": assay},
        "groups": {"type": "ARRAY", "items": group},
        "samples": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "pattern_or_sample": s(), "label": s(), "is_study_sample": b(),
            "confidence": n(), "evidence": s()}, "required": ["pattern_or_sample", "label", "is_study_sample"]}},
        "clarifying_questions": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "group_id": s(nullable=True), "question": s()}, "required": ["question"]}},
    }, "required": ["layout", "assays", "groups"]}


# ---------------------------------------------------------------- parsing (Pydantic)

class Fact(BaseModel):
    value: str
    confidence: float = 0.0
    evidence: str = ""


class FeatureIdentity(BaseModel):
    group_ids: List[str] = []
    composite: Optional[bool] = False


class Assay(BaseModel):
    assay_label: str
    omics_type: str = "unknown"
    source_software: Optional[str] = "unknown"
    in_supported_scope: str = "unsure"
    scope_reason: Optional[str] = ""
    feature_identity: Optional[FeatureIdentity] = None
    confidence: float = 0.0
    evidence: str = ""


class GroupLabel(BaseModel):
    group_id: str
    role: str
    assay_label: Optional[str] = None
    label: Optional[str] = ""
    block_role: Optional[str] = None
    audit_kind: Optional[str] = None
    marks_rows_as_suspect: Optional[bool] = None
    confidence: float = 0.0
    evidence: str = ""
    suggest_split: Optional[List[str]] = None
    suggest_split_role: Optional[str] = None
    suggest_split_audit_kind: Optional[str] = None
    suggest_split_label: Optional[str] = None


class SampleRule(BaseModel):
    pattern_or_sample: str
    label: str = ""
    is_study_sample: bool = True
    confidence: float = 0.0
    evidence: str = ""


class Question(BaseModel):
    group_id: Optional[str] = None
    question: str


class AIResponse(BaseModel):
    layout: Optional[Fact] = None
    assays: List[Assay] = []
    groups: List[GroupLabel] = []
    samples: List[SampleRule] = []
    clarifying_questions: List[Question] = []


def _clamp(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------- calling

def _cache_key(sha, model_key, digest):
    return hashlib.sha256(f"{sha}|{PROMPT_VERSION}|{model_key}|{digest_hash(digest)}".encode()).hexdigest()[:40]


def ask(digest, sha, log=None, mock_fn=None):
    """One validated-JSON round trip with a single retry on invalid output.
    Returns (AIResponse | None, meta). meta has provider, model, error, raw."""
    ok, why = llm.available()
    meta = {"provider": llm.provider_name(), "model": llm.model_list()[0] if ok else None,
            "prompt_version": PROMPT_VERSION, "temperature": llm.TEMPERATURE,
            "digest_hash": digest_hash(digest), "cached": False, "error": None}
    if not ok:
        meta["error"] = why
        return None, meta
    key = _cache_key(sha, f"{meta['provider']}:{','.join(llm.model_list())}", digest)
    cache_file = CACHE_DIR / f"{key}.json"
    prompt = "Digest (JSON):\n" + json.dumps(digest, ensure_ascii=False)
    if log:
        log("ai_request", {"provider": meta["provider"], "models": llm.model_list(),
                           "prompt_version": PROMPT_VERSION, "temperature": llm.TEMPERATURE,
                           "digest_hash": meta["digest_hash"], "digest": digest})
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            parsed = AIResponse.model_validate_json(cached["raw"])
            meta.update(model=cached.get("model"), cached=True, raw=cached["raw"])
            llm.say(f"using cached proposal ({cache_file.name})")
            if log:
                log("ai_response_raw", {"cached": True, "model": cached.get("model"), "raw": cached["raw"]})
            return parsed, meta
        except (ValueError, KeyError, ValidationError):
            pass
    last_err = None
    for attempt in (1, 2):
        try:
            raw, m = llm.complete_json(system_prompt(), prompt, response_schema(), mock_fn)
        except llm.LLMError as e:
            meta["error"] = str(e)
            llm.say(f"FAILED: {e}")
            if log:
                log("ai_response_raw", {"error": str(e)})
            return None, meta
        except Exception as e:
            traceback.print_exc()
            meta["error"] = f"Unexpected error: {type(e).__name__}: {e}"
            return None, meta
        meta.update(model=m["model"], finish_reason=m["finish_reason"], raw=raw)
        if log:
            log("ai_response_raw", {"attempt": attempt, "model": m["model"], "finish_reason": m["finish_reason"],
                                    "raw": raw})
        try:
            if m["finish_reason"] != "STOP":
                raise ValueError(f"finish_reason={m['finish_reason']}")
            parsed = AIResponse.model_validate_json(raw)
        except (ValueError, ValidationError) as e:
            last_err = f"invalid JSON from the model ({str(e).splitlines()[0][:200]})"
            llm.say(f"attempt {attempt}: {last_err}" + (" -> retrying once" if attempt == 1 else ""))
            continue
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"model": m["model"], "raw": raw}), encoding="utf-8")
        except OSError:
            pass
        return parsed, meta
    meta["error"] = last_err
    return None, meta


# ---------------------------------------------------------------- validate & merge

GROUP_KEYS = ("role", "assay_label", "label", "block_role", "audit_kind", "marks_rows_as_suspect")


def validate_response(resp, groups, cols, layout=None):
    """AIResponse -> proposal. Unknown group ids are rejected; values outside
    the closed sets and contradicted claims become unresolved."""
    by_id = {g["group_id"]: g for g in groups}
    out = {"groups": {}, "rejected": [], "samples": [], "clarifying_questions": [], "splits": [], "assays": []}
    if resp.layout is not None:
        out["layout"] = {"value": resp.layout.value, "confidence": _clamp(resp.layout.confidence),
                         "evidence": resp.layout.evidence, "source": "ai"}
        if resp.layout.value not in VOCABULARY["layout"]:
            out["layout"].update(value=UNRESOLVED, validation={
                "status": "contradicted", "messages": [f"'{resp.layout.value}' is not an allowed layout."]})
    layout = layout or (out.get("layout") or {}).get("value")
    for a in resp.assays:
        fi = a.feature_identity or FeatureIdentity()
        bad = [g for g in fi.group_ids if g not in by_id]
        item = {"assay_label": a.assay_label.strip() or "assay", "omics_type": a.omics_type or "unknown",
                "source_software": a.source_software or "unknown",
                "in_supported_scope": a.in_supported_scope if a.in_supported_scope in VOCABULARY["in_supported_scope"]
                else "unsure", "scope_reason": a.scope_reason or "",
                "feature_group_ids": [g for g in fi.group_ids if g in by_id],
                "confidence": _clamp(a.confidence), "evidence": a.evidence, "source": "ai"}
        if bad:
            out["rejected"].extend({"group_id": g, "reason": "feature identity names a group that does not exist"}
                                   for g in bad)
        out["assays"].append(item)
    for gl in resp.groups:
        if gl.group_id not in by_id:
            out["rejected"].append({"group_id": gl.group_id,
                                    "reason": "This group id does not exist in the digest; the claim was rejected."})
            continue
        if gl.group_id in out["groups"]:
            continue
        item = {"role": gl.role, "assay_label": gl.assay_label, "label": (gl.label or "").strip(),
                "block_role": gl.block_role, "audit_kind": gl.audit_kind,
                "marks_rows_as_suspect": bool(gl.marks_rows_as_suspect),
                "confidence": _clamp(gl.confidence), "evidence": gl.evidence, "source": "ai"}
        out["groups"][gl.group_id] = finalize_item(item, by_id[gl.group_id], cols, layout)
        if gl.suggest_split:
            out["splits"].append({"group_id": gl.group_id, "columns": gl.suggest_split,
                                  "role": gl.suggest_split_role or UNRESOLVED, "audit_kind": gl.suggest_split_audit_kind,
                                  "label": gl.suggest_split_label or "", "evidence": gl.evidence})
    out["samples"] = [{"pattern_or_sample": x.pattern_or_sample, "label": x.label, "is_study_sample": x.is_study_sample,
                       "confidence": _clamp(x.confidence), "evidence": x.evidence, "source": "ai"} for x in resp.samples]
    out["clarifying_questions"] = [q.model_dump() for q in resp.clarifying_questions]
    return out


def normalize_item(item):
    """Drop fields that do not apply to the role (keeps the item consistent)."""
    role = item.get("role")
    if role != "value":
        item["block_role"] = None
        if role not in ("feature_annotation",):
            item["assay_label"] = item.get("assay_label") if role == "feature_id" else None
    if role != "sample_metadata":
        item["audit_kind"] = None
    if role != "feature_annotation":
        item["marks_rows_as_suspect"] = False
    return item


def finalize_item(item, group, cols, layout=None):
    """Structural validation; contradicted -> unresolved (the claim is kept for display)."""
    item = normalize_item(item)
    v = validate_group(item, group, cols, layout)
    item["validation"] = v
    if v["status"] == "contradicted":
        item["claimed"] = {k: item.get(k) for k in ("role", "block_role", "audit_kind", "label")}
        item["role"] = UNRESOLVED
    if item.get("role") == "sample_metadata" and item.get("audit_kind") == "timepoint" and group["n_columns"] == 1:
        item.setdefault("detail", timepoint_detail(cols.digests[group["indices"][0]]))
    return item


def unresolved_item(evidence, source="none"):
    return {"role": UNRESOLVED, "assay_label": None, "label": "", "block_role": None, "audit_kind": None,
            "marks_rows_as_suspect": False, "confidence": 0.0, "evidence": evidence, "source": source,
            "validation": {"status": "ok", "messages": []}}


def propose(filename, sha, groups, hints, cols, fixed=None, group_ids=None, log=None, on_progress=None,
            mock_fn=None, signature_hint=None):
    """Ask the AI about all (or some) groups, in batches for very wide files.
    Returns (proposal, meta, digests)."""
    targets = [g for g in groups if group_ids is None or g["group_id"] in group_ids]
    batches = [targets[i:i + GROUPS_PER_CALL] for i in range(0, len(targets), GROUPS_PER_CALL)] or [[]]
    merged = {"groups": {}, "rejected": [], "samples": [], "clarifying_questions": [], "splits": [], "assays": []}
    metas, digests = [], []
    fixed = dict(fixed or {})
    for n, batch in enumerate(batches, 1):
        if on_progress:
            on_progress(n - 1, len(batches), f"AI batch {n}/{len(batches)}: {len(batch)} column group(s)")
        digest = build_digest(filename, groups, hints, fixed, {g["group_id"] for g in batch}, signature_hint)
        digests.append(digest)
        _ctx_set(on_progress, n, len(batches))
        try:
            resp, meta = ask(digest, sha, log, mock_fn)
        finally:
            _ctx_clear()
        metas.append(meta)
        if resp is None:
            for g in batch:
                merged["groups"][g["group_id"]] = unresolved_item(
                    f"AI unavailable: {meta['error']}. Choose this role yourself.")
            continue
        part = validate_response(resp, groups, cols, fixed.get("layout"))
        if "layout" in part and "layout" not in merged:
            merged["layout"] = part["layout"]
            fixed.setdefault("layout", part["layout"]["value"])
        known = {a["assay_label"] for a in merged["assays"]}
        merged["assays"].extend(a for a in part["assays"] if a["assay_label"] not in known)
        if merged["assays"]:
            fixed["assays_so_far"] = [a["assay_label"] for a in merged["assays"]]
        for g in batch:
            merged["groups"][g["group_id"]] = part["groups"].get(g["group_id"]) or unresolved_item(
                "The AI returned no label for this group.", "ai")
        for k in ("rejected", "samples", "clarifying_questions", "splits"):
            merged[k].extend(part[k])
        if log:
            log("validation_result", {"batch": n, "groups": {gid: it["validation"] for gid, it in part["groups"].items()},
                                      "rejected": part["rejected"]})
    if on_progress:
        on_progress(len(batches), len(batches), "AI proposal received and checked against the data")
    meta = metas[0] if metas else {}
    if any(m.get("error") for m in metas):
        meta = dict(meta, error=next(m["error"] for m in metas if m.get("error")))
    meta["models_used"] = sorted({m.get("model") for m in metas if m.get("model")})
    return merged, meta, digests


def _ctx_set(on_progress, n, total):
    if on_progress:
        llm._ctx.report = lambda m: on_progress(n - 1, total, m)


def _ctx_clear():
    llm._ctx.report = None
