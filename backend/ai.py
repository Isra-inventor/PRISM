"""AI layer (spec 4): label pre-built column groups from a statistics digest.

What is sent: file-level layout hints and, per group, its name pattern, a few
column names and aggregated statistics. Never raw rows. Example values only
for low-cardinality text columns (<= 20 distinct values), and only when
AI_SEND_EXAMPLE_VALUES is on (default).

What comes back is parsed strictly (Pydantic), then every claim is checked
against the data (validation.py). Contradicted claims become 'unresolved';
unknown group ids are rejected; missing groups are unresolved.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import traceback
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ValidationError

from . import llm_providers as llm
from .schema import ALL_KINDS, DEFINITIONS, PROMPT_VERSION, UNRESOLVED, VOCABULARY
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
    d["profile"] = prof
    if g.get("sample_names"):
        d["sample_names_after_stripping_pattern"] = g["sample_names"][:SAMPLE_NAMES_IN_DIGEST]
    if g.get("split_from"):
        d["split_from"] = g["split_from"]
        d["split_reasons"] = g.get("split_reasons")
    return d


def build_digest(filename, groups, hints, fixed=None, group_ids=None):
    examples = send_examples()
    chosen = [g for g in groups if group_ids is None or g["group_id"] in group_ids]
    return {
        "file": {"name_extension": Path(filename).suffix.lower(), "layout_hints": hints},
        "already_confirmed": fixed or {},
        "groups": [_group_digest(g, examples) for g in chosen],
        "settings": {"example_values_sent": examples, "raw_rows_sent": False},
    }


def digest_hash(digest):
    return hashlib.sha256(json.dumps(digest, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- prompt

def _defs(keys):
    return "\n".join(f"- {k}: {DEFINITIONS[k]}" for k in keys if k in DEFINITIONS)


SYSTEM_PROMPT = f"""You help a scientist confirm the structure of a quantified omics table \
(proteomics, metabolomics, ...). The columns were already grouped by deterministic code. \
You receive a JSON digest: layout hints and, per group, its name pattern, some column names \
and computed statistics. You never see raw data rows.

Your task: label every group, and propose file-level facts.
- layout: one of {VOCABULARY['layout']}.
- omics_type: one of {VOCABULARY['omics_type']}.
- source_software: the software that produced the file if the column names reveal it, else "unknown".
- feature_identity: the group_id(s) whose values identify each feature (several = composite key, \
e.g. m/z + retention time). Use [] when feature names are column headers (samples in rows).
- groups: for EVERY group in the digest, exactly one entry with:
  role: one of {VOCABULARY['column_role']}
  kind: for role feature_annotation one of {VOCABULARY['feature_annotation_kind']}; \
for role sample_metadata one of {VOCABULARY['sample_metadata_kind']}; otherwise null.
  detail: for kind timepoint one of {VOCABULARY['timepoint_detail']}; otherwise null.
  measurement_type (role value only): one of {VOCABULARY['measurement_type']}.
  scale (role value only): one of {VOCABULARY['scale']}.
  omics_type (role value only): omics type of this block (a file can hold several assays).
  label (role value only): short human name, e.g. "LFQ intensity", "Peak area".
  confidence: 0-1, your probability that the entry is right.
  evidence: one sentence citing only facts present in the digest (names, statistics).
  suggest_split: null, or a list of column names inside a multi-column group that do not \
belong with the rest (e.g. a clinical covariate such as age or serum iron inside a block of \
metabolite columns), with suggest_split_role/suggest_split_kind for them. Code decides.
- sample_types: sample names or glob patterns (e.g. "QC_*") that are QC / pool / blank / \
calibrator samples, each with type from {VOCABULARY['sample_type']}.
- clarifying_questions: questions you cannot resolve from the digest.

Definitions:
{_defs(['feature_id', 'feature_annotation', 'value', 'sample_id', 'sample_metadata', 'ignore',
        'flag_decoy', 'flag_contaminant', 'flag_other', 'group', 'technical_numeric', 'timepoint',
        'covariate_numeric', 'samples_in_columns', 'samples_in_rows', 'long'])}

Rules:
- Use only the closed vocabularies above. When unsure, answer "unresolved" / "unknown" \
rather than guess. Never invent, merge or split groups; only label the given group_ids.
- Never decide which variable is the research outcome (use kind "group" only as a candidate).
- Never give preprocessing or statistical advice.
- Features usually outnumber samples. In a samples-in-columns table, per-sample column families \
are role value and single numeric columns describing features are feature_annotation. In a \
samples-in-rows table, single numeric columns are usually sample_metadata (covariates, technical \
values) and wide numeric blocks are role value.
- Respect everything listed under already_confirmed."""


def response_schema():
    s = lambda **k: dict(type="STRING", **k)
    n = lambda: {"type": "NUMBER"}
    fact = {"type": "OBJECT", "properties": {"value": s(), "confidence": n(), "evidence": s()},
            "required": ["value", "confidence", "evidence"]}
    group = {"type": "OBJECT", "properties": {
        "group_id": s(),
        "role": s(enum=VOCABULARY["column_role"]),
        "kind": s(enum=ALL_KINDS, nullable=True),
        "detail": s(enum=VOCABULARY["timepoint_detail"], nullable=True),
        "measurement_type": s(enum=VOCABULARY["measurement_type"], nullable=True),
        "scale": s(enum=VOCABULARY["scale"], nullable=True),
        "omics_type": s(enum=VOCABULARY["omics_type"], nullable=True),
        "label": s(nullable=True),
        "confidence": n(),
        "evidence": s(),
        "suggest_split": {"type": "ARRAY", "items": s(), "nullable": True},
        "suggest_split_role": s(enum=VOCABULARY["column_role"], nullable=True),
        "suggest_split_kind": s(enum=ALL_KINDS, nullable=True),
    }, "required": ["group_id", "role", "confidence", "evidence"]}
    return {"type": "OBJECT", "properties": {
        "layout": fact, "omics_type": fact, "source_software": fact,
        "feature_identity": {"type": "OBJECT", "properties": {
            "group_ids": {"type": "ARRAY", "items": s()}, "composite": {"type": "BOOLEAN"},
            "confidence": n(), "evidence": s()}, "required": ["group_ids", "confidence", "evidence"]},
        "groups": {"type": "ARRAY", "items": group},
        "sample_types": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "pattern_or_sample": s(), "type": s(enum=VOCABULARY["sample_type"]),
            "confidence": n(), "evidence": s()}, "required": ["pattern_or_sample", "type"]}},
        "clarifying_questions": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "group_id": s(nullable=True), "question": s()}, "required": ["question"]}},
    }, "required": ["groups"]}


# ---------------------------------------------------------------- parsing (Pydantic)

class Fact(BaseModel):
    value: str
    confidence: float = 0.0
    evidence: str = ""


class FeatureIdentity(BaseModel):
    group_ids: List[str] = []
    composite: Optional[bool] = False
    confidence: float = 0.0
    evidence: str = ""


class GroupLabel(BaseModel):
    group_id: str
    role: str
    kind: Optional[str] = None
    detail: Optional[str] = None
    measurement_type: Optional[str] = None
    scale: Optional[str] = None
    omics_type: Optional[str] = None
    label: Optional[str] = None
    confidence: float = 0.0
    evidence: str = ""
    suggest_split: Optional[List[str]] = None
    suggest_split_role: Optional[str] = None
    suggest_split_kind: Optional[str] = None


class SampleTypeRule(BaseModel):
    pattern_or_sample: str
    type: str
    confidence: float = 0.0
    evidence: str = ""


class Question(BaseModel):
    group_id: Optional[str] = None
    question: str


class AIResponse(BaseModel):
    layout: Optional[Fact] = None
    omics_type: Optional[Fact] = None
    source_software: Optional[Fact] = None
    feature_identity: Optional[FeatureIdentity] = None
    groups: List[GroupLabel] = []
    sample_types: List[SampleTypeRule] = []
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
            raw, m = llm.complete_json(SYSTEM_PROMPT, prompt, response_schema(), mock_fn)
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

def _fact(f, source):
    if f is None:
        return None
    return {"value": f.value, "confidence": _clamp(f.confidence), "evidence": f.evidence, "source": source}


def validate_response(resp, groups, cols, layout=None):
    """AIResponse -> proposal dict with per-item validation. Unknown group ids are
    rejected; contradicted items become unresolved; missing groups are unresolved."""
    by_id = {g["group_id"]: g for g in groups}
    layout = layout or (resp.layout.value if resp.layout else None)
    out = {"groups": {}, "rejected": [], "sample_types": [], "clarifying_questions": [], "splits": []}
    for key in ("layout", "omics_type", "source_software"):
        f = _fact(getattr(resp, key), "ai")
        if f:
            allowed = {"layout": VOCABULARY["layout"], "omics_type": VOCABULARY["omics_type"]}.get(key)
            if allowed and f["value"] not in allowed:
                f = {"value": UNRESOLVED, "confidence": 0.0, "source": "ai", "evidence": f["evidence"],
                     "validation": {"status": "contradicted",
                                    "messages": [f"'{f['value']}' is not an allowed {key}."]}}
            out[key] = f
    if resp.feature_identity is not None:
        fi = resp.feature_identity
        item = {"group_ids": fi.group_ids, "composite": bool(fi.composite or len(fi.group_ids) > 1),
                "confidence": _clamp(fi.confidence), "evidence": fi.evidence, "source": "ai"}
        item["validation"] = validate_feature_identity(item, by_id, cols, layout)
        if item["validation"]["status"] == "contradicted":
            item["group_ids"] = []
        out["feature_identity"] = item
    for gl in resp.groups:
        if gl.group_id not in by_id:
            out["rejected"].append({"group_id": gl.group_id,
                                    "reason": "This group id does not exist in the digest; the claim was rejected."})
            continue
        if gl.group_id in out["groups"]:
            continue
        item = {"role": gl.role, "kind": gl.kind, "detail": gl.detail, "measurement_type": gl.measurement_type,
                "scale": gl.scale, "omics_type": gl.omics_type, "label": gl.label,
                "confidence": _clamp(gl.confidence), "evidence": gl.evidence, "source": "ai"}
        if gl.role != "value":
            for k in ("measurement_type", "scale", "label"):
                item[k] = None
        out["groups"][gl.group_id] = finalize_item(item, by_id[gl.group_id], cols, layout)
        if gl.suggest_split:
            out["splits"].append({"group_id": gl.group_id, "columns": gl.suggest_split,
                                  "role": gl.suggest_split_role or UNRESOLVED, "kind": gl.suggest_split_kind,
                                  "evidence": gl.evidence})
    out["sample_types"] = [{"pattern_or_sample": s.pattern_or_sample, "type": s.type,
                            "confidence": _clamp(s.confidence), "evidence": s.evidence, "source": "ai"}
                           for s in resp.sample_types if s.type in VOCABULARY["sample_type"]]
    out["clarifying_questions"] = [q.model_dump() for q in resp.clarifying_questions]
    return out


def finalize_item(item, group, cols, layout=None):
    """Validate one group item; contradicted -> unresolved (claim kept for display)."""
    if item.get("role") == "sample_metadata" and item.get("kind") == "timepoint" and group["n_columns"] == 1:
        auto = timepoint_detail(cols.digests[group["indices"][0]])
        if not item.get("detail") and auto:
            item["detail"] = auto
    v = validate_group(item, group, cols, layout)
    item["validation"] = v
    if v["status"] == "contradicted":
        item["claimed"] = {k: item.get(k) for k in ("role", "kind", "measurement_type", "scale")}
        item["role"] = UNRESOLVED
    return item


def unresolved_item(evidence, source="none"):
    return {"role": UNRESOLVED, "kind": None, "detail": None, "measurement_type": None, "scale": None,
            "omics_type": None, "label": None, "confidence": 0.0, "evidence": evidence, "source": source,
            "validation": {"status": "ok", "messages": []}}


def propose(filename, sha, groups, hints, cols, fixed=None, group_ids=None, log=None, on_progress=None,
            mock_fn=None):
    """Ask the AI about all (or some) groups, in batches for very wide files.
    Returns (proposal, meta, digests)."""
    targets = [g for g in groups if group_ids is None or g["group_id"] in group_ids]
    batches = [targets[i:i + GROUPS_PER_CALL] for i in range(0, len(targets), GROUPS_PER_CALL)] or [[]]
    merged = {"groups": {}, "rejected": [], "sample_types": [], "clarifying_questions": [], "splits": []}
    metas, digests = [], []
    fixed = dict(fixed or {})
    for n, batch in enumerate(batches, 1):
        if on_progress:
            on_progress(n - 1, len(batches), f"AI batch {n}/{len(batches)}: {len(batch)} column group(s)")
        digest = build_digest(filename, groups, hints, fixed, {g["group_id"] for g in batch})
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
        for key in ("layout", "omics_type", "source_software", "feature_identity"):
            if key in part and key not in merged:
                merged[key] = part[key]
                if key != "feature_identity":
                    fixed.setdefault(key, part[key]["value"])
        for g in batch:
            merged["groups"][g["group_id"]] = part["groups"].get(g["group_id"]) or unresolved_item(
                "The AI returned no label for this group.", "ai")
        for k in ("rejected", "sample_types", "clarifying_questions", "splits"):
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
