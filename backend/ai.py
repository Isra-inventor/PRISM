"""AI layer: the AI groups the columns and labels the groups, from facts.

What is sent: file-level layout hints and, per column, its name, position and
computed statistics, plus the name parts it shares with other columns
(shared_prefix / shared_suffix: a fact, not a grouping). Never raw rows.
Example values only for low-cardinality text columns (<= 20 distinct values),
and only when AI_SEND_EXAMPLE_VALUES is on (default).

Grouping is judgment, so it is the AI's proposal (v2.2): which columns are one
family, what each group is. Wide files are sent in chunks of
config.GROUPING_CHUNK_SIZE columns (a prompt size limit) and a consolidation
call lets the AI join families split across chunks. Code only applies
proposals whose ids and column names exist (grouping.py) and puts every column
nobody mentioned into its own 'unresolved' group.

What comes back is parsed strictly (Pydantic) and checked structurally
(validation.py): values outside the closed sets and contradicted claims
become 'unresolved'.
"""

from __future__ import annotations

import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ValidationError

from . import grouping, llm_providers as llm
from .config import GROUPING_CHUNK_SIZE, SCOPE_DESCRIPTION
from .profiling import chunk_columns, common_pattern
from .schema import DEFINITIONS, PROMPT_VERSION, UNRESOLVED, VOCABULARY
from .validation import timepoint_detail, validate_feature_identity, validate_group

CACHE_DIR = Path(os.environ.get("PRISM_CACHE_DIR", Path(__file__).parent / "cache"))


def send_examples():
    return os.environ.get("AI_SEND_EXAMPLE_VALUES", "true").strip().lower() not in ("0", "false", "no", "off")


# ---------------------------------------------------------------- digest

def column_digest(cols, i, affix, examples):
    d = dict(cols.digests[i])
    if not examples or (d.get("unique_ratio") or 0) >= 0.5:
        # identifier-like columns (e.g. patient codes) never send example values
        d.pop("values", None)
    if not examples:
        d.pop("value_shapes", None)
    d["shared_prefix"] = affix.get("shared_prefix")
    d["shared_suffix"] = affix.get("shared_suffix")
    return d


def build_digest(filename, cols, affixes, hints, indices, fixed=None, signature_hint=None, chunk=None):
    examples = send_examples()
    d = {
        "file": {"name_extension": Path(filename).suffix.lower(), "layout_hints": hints,
                 "signature_hint": signature_hint, "n_columns_in_file": len(cols.header)},
        "already_confirmed": fixed or {},
        "columns": [column_digest(cols, i, affixes[i], examples) for i in indices],
        "settings": {"example_values_sent": examples, "raw_rows_sent": False},
    }
    if chunk:
        d["chunk"] = {"index": chunk[0], "of": chunk[1], "n_columns": len(indices),
                      "note": ("Only these columns are in this call. Group them; a separate step joins "
                               "families that were split across chunks.")}
    return d


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
- groups: YOU group the columns. A group is a set of columns that are one kind of thing: one
  measurement taken for many samples (e.g. "LFQ intensity S01" ... "LFQ intensity S24"), one
  block of features measured in every sample, or a single annotation / sample column (a group
  of one). Give each group your own short group_id (g1, g2, ...) and list its exact column
  names in `columns`. Every column in the digest must be in exactly one group; code turns a
  column you leave out into an unresolved group of its own.
  shared_prefix / shared_suffix only say which literal name parts a column shares with others;
  they are hints, not groups. Names without shared text (Pt003_visit1, 004-w1, Glucose,
  Lactate) can still be one family when the statistics agree; columns with a shared prefix can
  still be different things. Do not put different kinds of thing in one group (e.g. a clinical
  covariate among metabolites, a total over all samples next to the per-sample columns).
  Put several columns in one group only when they are the SAME measurement or attribute
  repeated across samples (in samples-in-rows files: the same kind of feature measured in every
  sample). QC, blank, pool and calibrator samples of that measurement belong in the same group
  as the study samples; describe them in `samples`, not with separate groups (e.g. "QC_01 Peak
  area", "blank_01 Peak area" and "S01 Peak area" are ONE group: peak area for every sample). Every other column
  is a group of its own: do not bundle different annotations, identifiers or flag columns
  together, because each needs its own role, label and (for flags) its own flagged value.
  Role feature_id is only for the column(s) that identify each row; names, formulas and database
  IDs of a feature are feature_annotation. Value groups name their assay_label.
- propose_merge: only if two of your own groups turn out to be one family.
- suggest_split (with suggest_split_role / _audit_kind / _label): columns inside a group that
  are a different kind of thing; used when you are asked to reconsider an existing group.
- assays: one entry per assay (a file can hold several, e.g. proteins and metabolites side by
  side). feature_identity lists YOUR group_ids whose values identify each feature (several =
  composite key such as m/z + retention time); use [] when feature names are column headers.
- samples: sample names or glob patterns (e.g. "QC_*") with a label and is_study_sample.
- clarifying_questions for anything you cannot resolve from the digest.
- literature_queries: 1-3 Europe PMC search queries that would find papers analysing data like
  this (the platform / software, the measurement types, the organism or sample type if the names
  show it), e.g. "LFQ intensity" AND iBAQ AND MaxQuant. Names and terms only, never data values.
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
        "columns": {"type": "ARRAY", "items": s()},
        "role": s(enum=VOCABULARY["column_role"]),
        "assay_label": s(nullable=True),
        "label": s(),
        "audit_kind": s(enum=VOCABULARY["audit_kind"], nullable=True),
        "marks_rows_as_suspect": b(nullable=True),
        "confidence": n(),
        "evidence": s(),
        "suggest_split": {"type": "ARRAY", "items": s(), "nullable": True},
        "suggest_split_role": s(enum=VOCABULARY["column_role"], nullable=True),
        "suggest_split_audit_kind": s(enum=VOCABULARY["audit_kind"], nullable=True),
        "suggest_split_label": s(nullable=True),
    }, "required": ["group_id", "columns", "role", "label", "confidence", "evidence"]}
    return {"type": "OBJECT", "properties": {
        "layout": fact,
        "assays": {"type": "ARRAY", "items": assay},
        "groups": {"type": "ARRAY", "items": group},
        "samples": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "pattern_or_sample": s(), "label": s(), "is_study_sample": b(),
            "confidence": n(), "evidence": s()}, "required": ["pattern_or_sample", "label", "is_study_sample"]}},
        "clarifying_questions": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "group_id": s(nullable=True), "question": s()}, "required": ["question"]}},
        "literature_queries": {"type": "ARRAY", "items": s()},
        "propose_merge": merges_schema(),
    }, "required": ["layout", "assays", "groups"]}


def merges_schema():
    return {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "group_ids": {"type": "ARRAY", "items": {"type": "STRING"}}, "reason": {"type": "STRING"}},
        "required": ["group_ids", "reason"]}}


CONSOLIDATION_PROMPT = """

## This call: which groups are one family?
You get a compact summary of column groups (id, size, role, label, first / last column names,
the literal name part the group's columns share, one line of statistics). Return, in
cross_chunk_merges, the sets of group_ids that are really one family (e.g. the same
measurement for different samples, proposed separately because the file was sent in chunks).
Only merge groups that are the same kind of thing; when unsure, do not merge. Explain each
merge in reason. If a user_hint is given, the user thinks the listed groups are the same thing:
check it against the summaries and say in comment why you agree or not."""


def consolidation_schema():
    return {"type": "OBJECT", "properties": {"cross_chunk_merges": merges_schema(), "comment": {"type": "STRING"}},
            "required": ["cross_chunk_merges"]}


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
    columns: List[str] = []
    role: str
    assay_label: Optional[str] = None
    label: Optional[str] = ""
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


class Merge(BaseModel):
    group_ids: List[str] = []
    reason: str = ""


class ConsolidationResponse(BaseModel):
    cross_chunk_merges: List[Merge] = []
    comment: Optional[str] = ""


class AIResponse(BaseModel):
    propose_merge: List[Merge] = []
    layout: Optional[Fact] = None
    assays: List[Assay] = []
    groups: List[GroupLabel] = []
    samples: List[SampleRule] = []
    clarifying_questions: List[Question] = []
    literature_queries: List[str] = []


def _clamp(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------- calling

def _cache_key(sha, model_key, digest, kind="digest"):
    return hashlib.sha256(f"{sha}|{PROMPT_VERSION}|{kind}|{model_key}|{digest_hash(digest)}".encode()).hexdigest()[:40]


def ask(digest, sha, log=None, mock_fn=None):
    return _call(digest, sha, log, mock_fn)


def _call(digest, sha, log=None, mock_fn=None, kind="digest"):
    """One validated-JSON round trip with a single retry on invalid output; cached on
    disk by file, prompt version, models and the exact payload (so the chunk contents).
    Returns (parsed | None, meta). meta has provider, model, error, raw."""
    header, system, schema, model_cls = {
        "digest": ("Digest (JSON):", system_prompt(), response_schema(), AIResponse),
        "consolidation": ("Consolidation (JSON):", BRIEFING + CONSOLIDATION_PROMPT, consolidation_schema(),
                          ConsolidationResponse),
    }[kind]
    ok, why = llm.available()
    meta = {"provider": llm.provider_name(), "model": llm.model_list()[0] if ok else None,
            "prompt_version": PROMPT_VERSION, "temperature": llm.TEMPERATURE,
            "digest_hash": digest_hash(digest), "cached": False, "error": None}
    if not ok:
        meta["error"] = why
        return None, meta
    key = _cache_key(sha, f"{meta['provider']}:{','.join(llm.model_list())}", digest, kind)
    cache_file = CACHE_DIR / f"{key}.json"
    prompt = header + "\n" + json.dumps(digest, ensure_ascii=False)
    if log:
        log("ai_request", {"kind": kind, "provider": meta["provider"], "models": llm.model_list(),
                           "prompt_version": PROMPT_VERSION, "temperature": llm.TEMPERATURE,
                           "digest_hash": meta["digest_hash"], "digest": digest})
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            parsed = model_cls.model_validate_json(cached["raw"])
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
            raw, m = llm.complete_json(system, prompt, schema, mock_fn)
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
            parsed = model_cls.model_validate_json(raw)
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


# ---------------------------------------------------------------- validate & assemble

GROUP_KEYS = ("role", "assay_label", "label", "audit_kind", "marks_rows_as_suspect")


def _item(gl):
    return {"role": gl.role, "assay_label": gl.assay_label, "label": (gl.label or "").strip(),
            "audit_kind": gl.audit_kind, "marks_rows_as_suspect": bool(gl.marks_rows_as_suspect),
            "confidence": _clamp(gl.confidence), "evidence": gl.evidence, "source": "ai"}


def _split_item(gl, evidence):
    return {"role": gl.suggest_split_role or UNRESOLVED, "assay_label": gl.assay_label, "audit_kind": gl.suggest_split_audit_kind,
            "label": gl.suggest_split_label or "", "marks_rows_as_suspect": False, "confidence": 0.7, "source": "ai",
            "evidence": f"Split out of the group on the AI's suggestion: {evidence}"}


def read_chunk(resp, indices, label_to_idx, ns, out):
    """Validate one chunk's answer into out (layout, assays, pgroups, ...)."""
    rej = out["rejected"]
    if resp.layout is not None and "layout" not in out:
        out["layout"] = {"value": resp.layout.value, "confidence": _clamp(resp.layout.confidence),
                         "evidence": resp.layout.evidence, "source": "ai"}
        if resp.layout.value not in VOCABULARY["layout"]:
            out["layout"].update(value=UNRESOLVED, validation={
                "status": "contradicted", "messages": [f"'{resp.layout.value}' is not an allowed layout."]})
    groups = grouping.collect([(gl.group_id, gl.columns, _item(gl)) for gl in resp.groups],
                              set(indices), label_to_idx, ns, rej)
    for gl in resp.groups:
        if gl.suggest_split and f"{gl.group_id}{ns}" in groups:
            new = grouping.apply_split(groups, f"{gl.group_id}{ns}", gl.suggest_split, label_to_idx,
                                       lambda i, gl=gl: _split_item(gl, gl.evidence), rej)
            if new:
                out["splits_applied"].append({"group": gl.group_id, "new": new})
    for m in resp.propose_merge:
        keep = grouping.apply_merge(groups, [f"{g}{ns}" for g in m.group_ids], m.reason, rej)
        if keep:
            out["merges_applied"].append({"group_ids": m.group_ids, "into": keep, "reason": m.reason})
            for g in m.group_ids:
                out["alias"][f"{g}{ns}"] = keep
    out["groups"].update(groups)
    known = {a["assay_label"] for a in out["assays"]}
    for a in resp.assays:
        if (a.assay_label.strip() or "assay") in known:
            continue
        fi = a.feature_identity or FeatureIdentity()
        out["assays"].append({
            "assay_label": a.assay_label.strip() or "assay", "omics_type": a.omics_type or "unknown",
            "source_software": a.source_software or "unknown",
            "in_supported_scope": a.in_supported_scope if a.in_supported_scope in VOCABULARY["in_supported_scope"]
            else "unsure", "scope_reason": a.scope_reason or "",
            "feature_pids": [f"{g}{ns}" for g in fi.group_ids],
            "confidence": _clamp(a.confidence), "evidence": a.evidence, "source": "ai"})
    out["samples"] += [{"pattern_or_sample": x.pattern_or_sample, "label": x.label, "is_study_sample": x.is_study_sample,
                        "confidence": _clamp(x.confidence), "evidence": x.evidence, "source": "ai"} for x in resp.samples]
    out["clarifying_questions"] += [dict(q.model_dump(), group_id=f"{q.group_id}{ns}" if q.group_id else None)
                                    for q in resp.clarifying_questions]
    out["literature_queries"] += [q.strip()[:200] for q in resp.literature_queries
                                  if q and q.strip() and q.strip()[:200] not in out["literature_queries"]][:3]


def group_summary(cols, pid, g, chunk=None):
    """One compact line of facts per group, for the consolidation / merge-check call."""
    idx = sorted(g["indices"])
    names = [cols.labels[i] for i in idx]
    it = g["item"]
    st = [cols.digests[i] for i in idx]
    num = [d for d in st if d["type"] == "numeric"]
    meds = sorted(d.get("median") for d in num if d.get("median") is not None)
    pat = common_pattern([cols.header[i] for i in idx])
    stats = ("text" if not num else
             f"median of column medians {meds[len(meds) // 2] if meds else None}, "
             f"whole numbers {'yes' if all(d.get('integer_valued') for d in num) else 'no'}, "
             f"zeros {round(100 * sum(d.get('frac_zero') or 0 for d in num) / len(num))}%")
    return {"group_id": pid, "chunk": chunk, "n_columns": len(idx), "role": it.get("role"), "label": it.get("label"),
            "assay_label": it.get("assay_label"), "first_columns": names[:2], "last_column": names[-1],
            "shared_name_part": pat["text"] if pat else None, "stats": stats}


def normalize_item(item):
    """Drop fields that do not apply to the role (keeps the item consistent)."""
    role = item.get("role")
    if role != "value":
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
        item["claimed"] = {k: item.get(k) for k in ("role", "audit_kind", "label")}
        item["role"] = UNRESOLVED
    if item.get("role") == "sample_metadata" and item.get("audit_kind") == "timepoint" and group["n_columns"] == 1:
        if not item.get("detail"):
            item["detail"] = timepoint_detail(cols.digests[group["indices"][0]])
    return item


def unresolved_item(evidence, source="none"):
    return {"role": UNRESOLVED, "assay_label": None, "label": "", "audit_kind": None,
            "marks_rows_as_suspect": False, "confidence": 0.0, "evidence": evidence, "source": source,
            "validation": {"status": "ok", "messages": []}}


def propose(filename, sha, cols, affixes, hints, indices=None, fixed=None, log=None, on_progress=None,
            mock_fn=None, signature_hint=None, chunk_size=None):
    """Ask the AI to group and label the given columns (all by default).
    Wide sets go in chunks, then one consolidation call. Returns (proposal, meta, digests):
    proposal["groups"] is pid -> {"indices", "item", "origin"}, covering every column exactly once."""
    indices = list(range(len(cols.header))) if indices is None else sorted(indices)
    size = chunk_size or GROUPING_CHUNK_SIZE
    chunks = chunk_columns(indices, cols.labels, affixes, size)
    n = len(chunks)
    label_to_idx = {lab: i for i, lab in enumerate(cols.labels)}
    out = {"groups": {}, "rejected": [], "samples": [], "clarifying_questions": [], "assays": [],
           "literature_queries": [], "splits_applied": [], "merges_applied": [], "alias": {}, "chunks": n,
           "failed_chunks": [], "consolidation_error": None}
    metas, digests, chunk_of = [], [], {}
    fixed = dict(fixed or {})
    total = n + (1 if n > 1 else 0)
    for k, chunk in enumerate(chunks, 1):
        ns = f"_chunk{k}" if n > 1 else ""
        if on_progress:
            on_progress(k - 1, total, f"AI call {k}/{total}: grouping and labelling {len(chunk)} column(s)")
        digest = build_digest(filename, cols, affixes, hints, chunk, fixed, signature_hint, (k, n) if n > 1 else None)
        digests.append(digest)
        _ctx_set(on_progress, k, total)
        try:
            resp, meta = ask(digest, sha, log, mock_fn)
        finally:
            _ctx_clear()
        metas.append(meta)
        if resp is None:
            out["failed_chunks"].append(k)
            for i in chunk:
                out["groups"][f"u{i}{ns}"] = {"indices": [i], "origin": "ai_unavailable", "item": unresolved_item(
                    f"AI unavailable: {meta['error']}. Choose this column's role yourself.")}
            continue
        before = set(out["groups"])
        read_chunk(resp, chunk, label_to_idx, ns, out)
        for pid in set(out["groups"]) - before:
            chunk_of[pid] = k
        if "layout" in out:
            fixed.setdefault("layout", out["layout"]["value"])
        if out["assays"]:
            fixed["assays_so_far"] = [a["assay_label"] for a in out["assays"]]
    if n > 1 and any(m.get("error") is None for m in metas):
        if on_progress:
            on_progress(n, total, f"AI call {total}/{total}: joining families split across {n} chunks")
        payload = {"groups": [group_summary(cols, pid, g, chunk_of.get(pid)) for pid, g in out["groups"].items()],
                   "already_confirmed": {"layout": fixed.get("layout")}}
        digests.append(payload)
        _ctx_set(on_progress, total, total)
        try:
            cresp, cmeta = _call(payload, sha, log, mock_fn, kind="consolidation")
        finally:
            _ctx_clear()
        metas.append(cmeta)
        out["consolidation_error"] = cmeta.get("error") if cresp is None else None
        for m in (cresp.cross_chunk_merges if cresp else []):
            keep = grouping.apply_merge(out["groups"], [out["alias"].get(g, g) for g in m.group_ids], m.reason,
                                        out["rejected"], where="consolidation")
            if keep:
                out["merges_applied"].append({"group_ids": m.group_ids, "into": keep, "reason": m.reason,
                                              "cross_chunk": True})
                for g in m.group_ids:
                    out["alias"][g] = keep
    if log:
        log("grouping_result", {"chunks": n, "n_groups": len(out["groups"]), "merges": out["merges_applied"],
                                "splits": out["splits_applied"], "rejected": out["rejected"]})
    if on_progress:
        on_progress(total, total, "AI proposal received and checked against the data")
    meta = metas[0] if metas else {}
    if any(m.get("error") for m in metas):
        meta = dict(meta, error=next(m["error"] for m in metas if m.get("error")))
    meta["models_used"] = sorted({m.get("model") for m in metas if m.get("model")})
    meta["calls"] = len(metas)
    meta["all_failed"] = bool(metas) and all(m.get("error") for m in metas[:n])
    return out, meta, digests


def consolidate(cols, groups, items, sha, log=None, mock_fn=None):
    """The consolidation call on the current groups (e.g. to retry it after a failure).
    Returns (merges [(group_ids, reason)], error). Nothing is applied here."""
    payload = {"groups": [group_summary(cols, gid, {"indices": g["indices"], "item": items[gid]})
                          for gid, g in groups.items()]}
    resp, meta = _call(payload, sha, log, mock_fn, kind="consolidation")
    if resp is None:
        return [], meta.get("error") or "no answer"
    return [(m.group_ids, m.reason) for m in resp.cross_chunk_merges], None


def merge_check(cols, groups, items, gids, hint, sha, log=None, mock_fn=None):
    """Ask the AI whether the user's 'these are the same thing' holds. Nothing is applied here."""
    payload = {"groups": [group_summary(cols, gid, {"indices": groups[gid]["indices"], "item": items[gid]})
                          for gid in gids],
               "user_hint": hint or "The user thinks these groups are the same thing."}
    resp, meta = _call(payload, sha, log, mock_fn, kind="consolidation")
    if resp is None:
        return {"agrees": None, "reason": meta.get("error") or "no answer", "comment": ""}
    want = set(gids)
    hit = next((m for m in resp.cross_chunk_merges if want <= set(m.group_ids)), None)
    return {"agrees": hit is not None, "reason": hit.reason if hit else "", "comment": resp.comment or ""}


def _ctx_set(on_progress, n, total):
    if on_progress:
        llm._ctx.report = lambda m: on_progress(n - 1, total, m)


def _ctx_clear():
    llm._ctx.report = None
