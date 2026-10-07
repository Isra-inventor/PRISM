"""Session workflow: build the draft schema, apply wizard steps, track provenance.

The draft is the single source of truth for a session. The frontend renders
it; every wizard step sends a decision that is applied here, validated and
logged. Provenance is derived by comparing the current value with the value
first proposed and who proposed it:
    computed | ai_proposed_confirmed | ai_proposed_corrected | user_set
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import threading
import uuid
from collections import Counter, OrderedDict
from pathlib import Path

from . import accounting, ai, consistency, derive, design, grouping, llm_providers as llm, questions
from .errors import StepError
from .format_detect import signature_hint, signature_prefill
from .mock_llm import expand_sample_rules
from .parsing import cell, is_missing, parse_bytes, sanitize_filename
from .profiling import Columns, apply_rule, layout_hints, make_group, shared_affixes
from .schema import HISTORY_QUESTIONS, SCHEMA_VERSION, UNRESOLVED, VOCABULARY
from .session_log import log_event
from .validation import timepoint_detail, validate_feature_identity, validate_group
from . import edits  # noqa: E402  (the edit layer registers ops defined below)

SESSIONS_DIR = Path(os.environ.get("PRISM_SESSIONS_DIR", Path(__file__).parent / "sessions"))
STEPS = ["layout", "feature_id", "annotations", "values", "samples", "sample_info", "design", "history", "review"]
GROUP_FIELDS = ("role", "assay_label", "label", "audit_kind", "marks_rows_as_suspect",
                "flagged_values", "detail", "keep", "family")
ASSAY_FIELDS = ("assay_label", "omics_type", "omics_family", "source_software", "in_supported_scope", "scope_reason")
FACT_FIELDS = ("value",)
FI_FIELDS = ("group_ids", "composite")
SAMPLE_FIELDS = ("label", "is_study_sample")
FLAG_MAX_VALUES = 5
_NON_STUDY = re.compile(r"(?i)(^|[^a-z])(qc|pool|pooled|blank|buffer|calib\w*|std|standard)([^a-z]|$)")
_FLAG_WORDS = {"+", "x", "yes", "y", "true", "1", "flag", "flagged", "reverse", "rev", "con"}
_MAX_SESSIONS = 20
_sessions = OrderedDict()
_lock = threading.Lock()


# ---------------------------------------------------------------- provenance

def provenance(item, fields):
    """Unchanged since proposed: computed / ai_proposed_confirmed. Changed: by whom (v2.4 §3):
    an AI patch or question option the user applied -> ai_proposed_confirmed (or _corrected
    when the user edited it first); anything the user set -> user_set."""
    src = item.get("source") or "none"
    proposed = item.get("proposed") or {}
    changed = any(item.get(f) != proposed.get(f) for f in fields if f in proposed or f in item)
    if not changed:
        return {"computed": "computed", "ai": "ai_proposed_confirmed"}.get(src, "user_set")
    if item.get("set_by") in ("ai_patch", "question_option"):
        return "ai_proposed_corrected" if item.get("set_corrected") else "ai_proposed_confirmed"
    return "user_set"


def _snap(item, fields):
    item["proposed"] = {f: copy.deepcopy(item.get(f)) for f in fields}
    return item


# ---------------------------------------------------------------- session

class Session:
    def __init__(self, sid, filename, raw):
        self.sid = sid
        self.filename = filename
        self.dir = SESSIONS_DIR / sid
        self.table = parse_bytes(filename, raw)
        self.sha = self.table["sha256"]
        self.cols = Columns(self.table)
        self.affixes = shared_affixes(self.cols.labels)   # facts about the names, not groups
        self.hints = layout_hints(self.cols)
        self.groups = []   # proposed by the AI (or a known format / you) in build_draft
        self.draft = None
        self.digests = []
        self.metadata_table = None
        self.lock = threading.RLock()
        self.study = None       # {"session_id", "dataset_id"} when this dataset belongs to a v3 session
        self.undo = []          # the edit layer's undo stack (edits.py), saved to undo.json
        self.undo_dirty = False
        self._tx = None

    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "state.json").write_text(json.dumps({
            "sid": self.sid, "filename": self.filename, "groups": group_structure(self), "draft": self.draft,
            "digests": self.digests, "study": self.study}, ensure_ascii=False), encoding="utf-8")
        if self.undo_dirty:
            (self.dir / "undo.json").write_text(json.dumps(self.undo, ensure_ascii=False), encoding="utf-8")
            self.undo_dirty = False

    def log(self, event, payload):
        log_event(self.sid, event, payload)

    @property
    def groups_by_id(self):
        # cached per groups list: every regroup assigns a new list, so identity is enough
        if getattr(self, "_gbi_src", None) is not self.groups:
            self._gbi = {g["group_id"]: g for g in self.groups}
            self._gbi_src = self.groups
        return self._gbi


def create_session(filename, raw):
    sid = uuid.uuid4().hex[:12]
    fname = sanitize_filename(filename)
    s = Session(sid, fname, raw)
    s.dir.mkdir(parents=True, exist_ok=True)
    (s.dir / ("source" + Path(fname).suffix.lower())).write_bytes(raw)
    _remember(s)
    return s


def _remember(s):
    with _lock:
        _sessions[s.sid] = s
        _sessions.move_to_end(s.sid)
        while len(_sessions) > _MAX_SESSIONS:
            _sessions.popitem(last=False)


def get_session(sid):
    if not re.fullmatch(r"[0-9a-f]{12}", sid or ""):
        raise KeyError(sid)
    with _lock:
        if sid in _sessions:
            return _sessions[sid]
    d = SESSIONS_DIR / sid
    state = d / "state.json"
    srcs = list(d.glob("source.*"))
    if not state.exists() or not srcs:
        raise KeyError(sid)
    st = json.loads(state.read_text(encoding="utf-8"))
    s = Session(sid, st["filename"], srcs[0].read_bytes())
    s.groups = [make_group(s.cols, g["group_id"], g["indices"], g.get("origin", "restored"),
                           **{k: g.get(k) for k in ("split_from", "merged_from")}) for g in st.get("groups", [])]
    s.draft, s.digests = st.get("draft"), st.get("digests", [])
    s.study = st.get("study")
    if s.draft:   # drafts saved before v2.4 stage 6
        s.draft["steps"].setdefault("design", "pending")
        s.draft.setdefault("design", _snap_design(design.blank(), "none"))
    if (d / "undo.json").exists():
        s.undo = json.loads((d / "undo.json").read_text(encoding="utf-8"))
    md = d / "metadata_source.csv"
    if md.exists() and s.draft and s.draft.get("metadata") and not s.draft["metadata"].get("skipped"):
        t = parse_bytes(s.draft["metadata"]["filename"], md.read_bytes())
        s.metadata_table = {"table": t, "cols": Columns(t)}
    _remember(s)
    return s


# ---------------------------------------------------------------- groups (proposed, never decided by code)

def group_structure(s):
    return [{k: g.get(k) for k in ("group_id", "indices", "origin", "split_from", "merged_from")} for g in s.groups]


def _next_id(s):
    nums = [int(g["group_id"][1:]) for g in s.groups if re.fullmatch(r"g\d+", g["group_id"])]
    return max(nums, default=0) + 1


def install_groups(s, pgroups, layout, keep_ids=None, first_id=1):
    """Turn proposed groups (pid -> {indices, item, origin}) into group records and
    validated items. Returns (groups, items, id_map)."""
    order, id_map = grouping.finalize(pgroups, first_id, keep_ids)
    groups, items = [], {}
    for pid in order:
        pg = pgroups[pid]
        gid = id_map[pid]
        g = make_group(s.cols, gid, pg["indices"], pg.get("origin", "ai_proposed"),
                       split_from=id_map.get(pg.get("split_from"), pg.get("split_from")),
                       merged_from=pg.get("merged_from"))
        groups.append(g)
        items[gid] = ai.finalize_item(_blank(dict(pg["item"])), g, s.cols, layout)
    return groups, items, id_map


def _singletons(s, evidence):
    return {f"u{i}": {"indices": [i], "origin": "manual", "item": dict(grouping.unmentioned_item(), evidence=evidence)}
            for i in range(len(s.cols.header))}


# ---------------------------------------------------------------- hints

def group_hint(g):
    p = g.get("profile") or {}
    pat = (g.get("pattern") or {}).get("text")
    if g["kind"] == "numeric_block":
        base = f"{g['n_columns']} numeric columns" + (f" sharing '{pat.strip()}'" if pat and pat.strip() else "")
        return base + f"; median {p.get('median')}, {int((p.get('frac_zero') or 0) * 100)}% zeros."
    if g["kind"] == "column_group":
        return f"{g['n_columns']} columns of mixed types" + (f" sharing '{pat.strip()}'" if pat and pat.strip() else "") + "."
    if g["type"] == "numeric":
        return (f"Numeric column: median {p.get('median')}, range {p.get('min')}–{p.get('max')}"
                + (", whole numbers" if p.get("integer_valued") else "") + ".")
    vals = p.get("values")
    if vals and (p.get("unique_ratio") or 0) < 0.5:
        return f"Text with {p.get('n_unique')} distinct values: " + ", ".join(v["value"] for v in vals[:6]) + "."
    return f"Text with {p.get('n_unique')} distinct values ({int((p.get('unique_ratio') or 0) * 100)}% unique)."


def layout_hint_text(h):
    if h["long_format_pattern"]:
        return "Repeated identifier columns and a single numeric column: this may be a long table."
    if h["numeric_columns"]:
        r = h["rows_to_numeric_columns_ratio"]
        if r is not None and r >= 1:
            return (f"{h['n_rows']} rows and {h['numeric_columns']} numeric columns: features usually outnumber "
                    "samples, so rows are probably features (samples in columns).")
        return (f"Only {h['n_rows']} rows but {h['numeric_columns']} numeric columns: "
                "rows are probably samples (samples in rows).")
    return "No numeric columns were found."


def flag_values(s, gid):
    """{value: count} for a single column with <= FLAG_MAX_VALUES distinct values (empty = ''), else None."""
    g = s.groups_by_id[gid]
    if g["n_columns"] != 1:
        return None
    i = g["indices"][0]
    c = Counter("" if is_missing(cell(r, i)) else cell(r, i).strip() for r in s.table["rows"])
    return dict(c.most_common()) if len(c) <= FLAG_MAX_VALUES else None


def _proposed_flag_value(values):
    if not values:
        return None
    nonempty = [v for v in values if v not in ("", "0", "false", "False", "no", "-")]
    if len(nonempty) == 1:
        return nonempty[0]
    words = [v for v in nonempty if v.lower() in _FLAG_WORDS]
    return words[0] if len(words) == 1 else None


# ---------------------------------------------------------------- building the draft

def _blank(item):
    base = {"role": UNRESOLVED, "assay_label": None, "label": "", "audit_kind": None,
            "marks_rows_as_suspect": False, "flagged_values": [], "detail": None, "keep": True, "family": None,
            "confidence": 0.0, "evidence": "", "source": "none", "validation": {"status": "ok", "messages": []}}
    base.update({k: v for k, v in item.items() if v is not None or k not in base})
    return base


def propose(s, ai_on=True, on_progress=None):
    """A fresh proposal: the session starts over, so the undo stack is cleared."""
    if s.draft and s.draft.get("finalized"):
        raise StepError("This dataset is finalized; its schema can no longer change.")
    s.undo, s.undo_dirty = [], True
    return build_draft(s, ai_on=ai_on, on_progress=on_progress)


def build_draft(s, ai_on=True, fixed_layout=None, on_progress=None):
    """AI proposal (signature appended as a hint) or, when the AI is off or
    unavailable, the signature pre-fill / manual hints -> a fresh draft."""
    hint = signature_hint(s.table["header"])
    ok, why = llm.available()
    use_ai = ai_on and ok
    prop, meta, digests = None, {"error": None}, []
    if use_ai:
        fixed = {"layout": fixed_layout} if fixed_layout else {}
        prop, meta, digests = ai.propose(s.filename, s.sha, s.cols, s.affixes, s.hints, fixed=fixed, log=s.log,
                                         on_progress=on_progress, signature_hint=hint)
        if meta.get("all_failed"):
            prop = None  # the AI failed entirely: fall back to manual starting points
    manual = prop is None
    if manual:
        prop = signature_prefill(s.table["header"], s.cols) or {
            "groups": _singletons(s, "AI is off: choose what this column is, and group columns that belong "
                                     "together (see 'Shared name parts').")}
    s.digests = digests
    layout0 = fixed_layout or (prop.get("layout") or {}).get("value")
    s.groups, items, id_map = install_groups(s, prop["groups"], layout0)
    alias = prop.get("alias", {})
    to_gid = lambda pid: id_map.get(alias.get(pid, pid)) if pid else None

    d = {"schema_version": SCHEMA_VERSION, "groups": {}, "sample_rules": {}, "samples": {}, "sample_rules_ai": [],
         "processing_history": {q: {"answer": None, "note": "", "provenance": "unanswered", "answered_at": None}
                                for q, _ in HISTORY_QUESTIONS},
         "software_and_version": "", "history_notes": "", "metadata": None,
         "steps": {st: "pending" for st in STEPS}, "signature_hint": hint}
    d["ai"] = {"enabled": bool(ai_on), "available": ok, "unavailable_reason": None if ok else why,
               "used": use_ai and not manual, "provider": meta.get("provider") or llm.provider_name(),
               "model": meta.get("model"), "models_used": meta.get("models_used", []),
               "prompt_version": ai.PROMPT_VERSION, "temperature": llm.TEMPERATURE, "error": meta.get("error"),
               "cached": meta.get("cached", False), "calls": meta.get("calls", 0)}
    d["grouping"] = {"source": "ai" if not manual else ("signature" if prop.get("signature") else "manual"),
                     "chunks": prop.get("chunks", 1), "merges_applied": prop.get("merges_applied", []),
                     "splits_applied": prop.get("splits_applied", []),
                     "chunk_size": ai.config.GROUPING_CHUNK_SIZE, "failed_chunks": prop.get("failed_chunks", []),
                     "chunk_mode": prop.get("chunk_mode"), "n_templates": prop.get("n_templates"),
                     "consolidation_error": prop.get("consolidation_error")}

    # layout
    if fixed_layout:
        lay = {"value": fixed_layout, "confidence": 1.0, "evidence": "Chosen by you.", "source": "user"}
    elif prop.get("layout"):
        lay = dict(prop["layout"])
    else:
        lay = {"value": UNRESOLVED, "confidence": 0.0, "evidence": "", "source": "none"}
    lay.setdefault("validation", {"status": "ok", "messages": []})
    lay["hint"] = layout_hint_text(s.hints)
    d["layout"] = _snap(lay, FACT_FIELDS)
    layout = lay["value"]

    # assays
    assays = [dict(a) for a in prop.get("assays", [])]
    if not assays:
        assays = [{"assay_label": "assay 1", "omics_type": "unknown", "omics_family": "unknown", "source_software": "unknown",
                   "in_supported_scope": "unsure", "scope_reason": "not described yet", "confidence": 0.0,
                   "evidence": "", "source": "none"}]
    for a in assays:
        a.setdefault("omics_family", "unknown")
    d["assays"] = [_snap(a, ASSAY_FIELDS) for a in assays]
    for q, hint in (prop.get("processing_hints") or {}).items():
        d["processing_history"][q]["ai_hint"] = hint

    # groups
    labels = [a["assay_label"] for a in d["assays"]]
    for g in s.groups:
        gid = g["group_id"]
        item = items[gid]
        item["hint"] = group_hint(g)
        if item["role"] == "value" and item.get("assay_label") not in labels:
            item["assay_label"] = labels[0]
        _attach_flags(s, gid, item)
        d["groups"][gid] = item
    if use_ai and not manual:
        _relabel_repeats(s, d)
    for gid, item in d["groups"].items():
        if item["role"] != UNRESOLVED:
            item["validation"] = validate_group(item, s.groups_by_id[gid], s.cols, layout)
        for w in item.pop("label_warning", None) or []:
            item["validation"]["messages"].append(w)
            if item["validation"]["status"] == "ok":
                item["validation"]["status"] = "warning"
        _snap(item, GROUP_FIELDS)

    # feature identity: first assay that names one, or the groups labelled feature_id
    for a in assays:
        a["feature_group_ids"] = [g for g in dict.fromkeys(to_gid(p) for p in a.pop("feature_pids", [])) if g]
    fi_ids = next((a.get("feature_group_ids") for a in assays if a.get("feature_group_ids")), None)
    if fi_ids:
        fi = {"group_ids": fi_ids, "composite": len(fi_ids) > 1, "confidence": assays[0].get("confidence", 0),
              "evidence": assays[0].get("evidence", ""), "source": assays[0].get("source", "ai")}
    elif prop.get("feature_identity"):
        fi = dict(prop["feature_identity"])
    else:
        ids = [gid for gid, it in d["groups"].items() if it["role"] == "feature_id"]
        fi = {"group_ids": ids, "composite": len(ids) > 1, "confidence": 0.5 if ids else 0.0,
              "evidence": "Groups labelled as feature ID." if ids else "", "source": "computed" if ids else "none"}
    for gid in fi["group_ids"]:
        if d["groups"][gid]["role"] in (UNRESOLVED, "feature_annotation"):
            d["groups"][gid]["role"] = d["groups"][gid]["proposed"]["role"] = "feature_id"
    fi["validation"] = validate_feature_identity(fi, s.groups_by_id, s.cols, layout)
    d["feature_identity"] = _snap(fi, FI_FIELDS)

    sid_groups = [gid for gid, it in d["groups"].items() if it["role"] == "sample_id"]
    d["sample_id_group"] = _snap({"value": sid_groups[0] if sid_groups else None,
                                  "source": d["groups"][sid_groups[0]]["source"] if sid_groups else "none"},
                                 FACT_FIELDS)
    for g in s.groups:
        if g.get("sample_id_rule") is not None:
            d["sample_rules"][g["group_id"]] = _snap(dict(g["sample_id_rule"], source="computed"),
                                                     ("strip_prefix", "strip_suffix"))
    d["sample_rules_ai"] = prop.get("samples", [])
    d["design"] = design_from_ai(s, d, prop.get("design"), "proposal") or _snap_design(design.blank(), "none")
    old = s.draft or {}
    d["answers"] = dict(old.get("answers") or {})          # answered / dismissed questions are never asked again
    d["questions"] = [q for q in old.get("questions", []) if questions.status(old, q) != "open"]
    d["chat"], d["patches"] = old.get("chat", []), [p for p in old.get("patches", []) if p["status"] == "applied"]
    d["settings"] = dict(old.get("settings") or {})
    d["rejected"] = prop.get("rejected", [])
    refresh_samples(s, d)
    s.draft = d
    for q in prop.get("clarifying_questions", []):
        questions.from_ai(s, d, dict(q, group_id=to_gid(q.get("group_id"))), "proposal")
    after_edit(s)
    s.log("proposal", {"ai": d["ai"], "signature_hint": hint, "layout": d["layout"], "assays": d["assays"],
                       "groups": {gid: dict({k: it.get(k) for k in GROUP_FIELDS + ("confidence", "source", "validation")},
                                            columns=s.groups_by_id[gid]["columns"])
                                  for gid, it in d["groups"].items()},
                       "questions": [q["text"] for q in d.get("questions", []) if q["source"] == "ai"],
                       "feature_identity": d["feature_identity"]})
    s.save()
    return d


LABEL_REPEAT_MAX = 5


def _relabel_repeats(s, d):
    """v2.4 §8.1: more than 5 annotation columns sharing one identical label get a warning and
    one retry asking the AI for column-specific labels."""
    by = {}
    for gid, it in d["groups"].items():
        g = s.groups_by_id[gid]
        if it["role"] == "feature_annotation" and g["n_columns"] == 1 and it.get("label"):
            by.setdefault(it["label"].strip().lower(), []).append(gid)
    retries = []
    for lab, gids in by.items():
        if len(gids) <= LABEL_REPEAT_MAX:
            continue
        idx = [s.groups_by_id[g]["indices"][0] for g in gids]
        shared = d["groups"][gids[0]]["label"]
        got, err = ai.relabel(s.cols, s.affixes, idx, shared, s.sha, log=s.log)
        fixed = 0
        for g, i in zip(gids, idx):
            x = (got or {}).get(s.cols.labels[i])
            it = d["groups"][g]
            if x:
                it["label"] = x["label"].strip()[:300]
                if x.get("family"):
                    it["family"] = str(x["family"]).strip()[:120] or None
                elif not it.get("family"):
                    it["family"] = shared
                fixed += 1
        after = Counter(d["groups"][g]["label"].strip().lower() for g in gids)
        for g in gids:
            n = after[d["groups"][g]["label"].strip().lower()]
            if n > LABEL_REPEAT_MAX:
                d["groups"][g]["label_warning"] = [f"{n} annotation columns share the label "
                                                   f"'{d['groups'][g]['label']}': describe each column specifically."]
        retries.append({"label": shared, "n_columns": len(gids), "relabelled": fixed, "error": err,
                        "still_shared": sum(1 for g in gids if d["groups"][g].get("label_warning"))})
    if retries:
        d["grouping"]["label_retries"] = retries
        s.log("label_retry", {"retries": retries})


def _attach_flags(s, gid, item):
    item["flag_values"] = flag_values(s, gid) if item.get("marks_rows_as_suspect") else None
    if item.get("marks_rows_as_suspect") and not item.get("flagged_values"):
        v = _proposed_flag_value(item["flag_values"])
        item["flagged_values"] = [v] if v is not None else []


# ---------------------------------------------------------------- samples

def layout_of(d):
    return d["layout"]["value"]


def value_blocks(s, d):
    """Kept value blocks, in file order. None is ranked above another."""
    return [gid for gid, it in d["groups"].items() if it["role"] == "value" and it.get("keep", True)]


def sample_ids(s, d, gid=None):
    """Sample IDs in file order, as the canonical outputs will name them."""
    lay = layout_of(d)
    if lay == "samples_in_columns":
        out = []
        for b in ([gid] if gid else value_blocks(s, d)):
            g = s.groups_by_id.get(b)
            if not g:
                continue
            rule = d["sample_rules"].get(b) or {"strip_prefix": "", "strip_suffix": ""}
            out.extend(n for n in (apply_rule(s.table["header"][i], rule) for i in g["indices"]) if n not in out)
        return out
    sg = (d.get("sample_id_group") or {}).get("value")
    if not sg or sg not in s.groups_by_id:
        return [f"row_{k + 1}" for k in range(len(s.table["rows"]))] if lay == "samples_in_rows" else []
    i = s.groups_by_id[sg]["indices"][0]
    vals = [cell(r, i).strip() for r in s.table["rows"]]
    if lay == "long":
        return list(OrderedDict.fromkeys(vals))
    return vals


def refresh_samples(s, d):
    """(Re)compute each sample's label and is_study_sample, keeping user edits."""
    ids = sample_ids(s, d)
    rules = expand_sample_rules(d.get("sample_rules_ai", []), ids)
    if layout_of(d) == "samples_in_columns":  # the AI may name raw column names instead of sample IDs
        for gid in value_blocks(s, d):
            rule = d["sample_rules"].get(gid) or {"strip_prefix": "", "strip_suffix": ""}
            for i in s.groups_by_id[gid]["indices"]:
                sid = apply_rule(s.table["header"][i], rule)
                if sid not in rules:
                    hit = expand_sample_rules(d.get("sample_rules_ai", []), [s.table["header"][i]])
                    if hit:
                        rules[sid] = next(iter(hit.values()))
    type_col = None
    if layout_of(d) == "samples_in_rows":
        for gid, it in d["groups"].items():
            if it["role"] == "sample_metadata" and it.get("audit_kind") == "sample_type":
                type_col = s.groups_by_id[gid]["indices"][0]
    col_vals = {}
    if type_col is not None:
        for r, sid in zip(s.table["rows"], ids):
            col_vals[sid] = cell(r, type_col).strip()
    old = d.get("samples", {})
    new = {}
    for sid in ids:
        if sid in old and old[sid].get("user_edited"):
            new[sid] = old[sid]
            continue
        v = col_vals.get(sid)
        if v and v in rules:  # the AI described this sample-type value
            r = rules[v]
            item = {"label": r["label"] or v, "is_study_sample": r["is_study_sample"], "source": "ai",
                    "evidence": f"Sample type column says '{v}'. {r['evidence']}", "confidence": r["confidence"]}
        elif v:
            item = {"label": v, "is_study_sample": not _NON_STUDY.search(v), "source": "computed",
                    "evidence": f"Sample type column says '{v}'.", "confidence": 0.6}
        elif sid in rules:
            r = rules[sid]
            item = {"label": r["label"], "is_study_sample": r["is_study_sample"], "source": "ai",
                    "evidence": r["evidence"], "confidence": r["confidence"]}
        elif _NON_STUDY.search(sid):
            m = _NON_STUDY.search(sid).group(2)
            item = {"label": m.lower(), "is_study_sample": False, "source": "computed",
                    "evidence": f"Sample name contains '{m}'.", "confidence": 0.6}
        else:
            item = {"label": "study sample", "is_study_sample": True, "source": "computed",
                    "evidence": "No QC / blank / pool / calibrator hint in the name.", "confidence": 0.5}
        new[sid] = _snap(item, SAMPLE_FIELDS)
    d["samples"] = new


# ---------------------------------------------------------------- confirming steps

def step_applicable(s, d, step):
    if step == "annotations":
        return layout_of(d) != "samples_in_rows" or any(it["role"] == "feature_annotation" for it in d["groups"].values())
    return True


def confirm_step(s, step, decision, on_progress=None):
    """Apply a step decision as one undoable edit: on any error the draft is left untouched."""
    if step not in STEPS:
        raise StepError(f"Unknown step '{step}'.")
    with edits.recording(s, "confirm_step", "user", f"Confirmed step {STEPS.index(step) + 1} ({step.replace('_', ' ')})",
                         {"step": step, "decision": decision}):
        reproposed = _confirm_step(s, step, decision, on_progress)
    s.save()
    return {"draft": public_draft(s), "reproposed": reproposed}


def _clean(text, limit=300):
    return (str(text) if text is not None else "").strip()[:limit]


def _confirm_step(s, step, decision, on_progress=None):
    def E(op_name, args, **kw):
        return edits.apply_edit(s, op_name, args, "user", step=step, **kw)
    d = s.draft
    before = json.loads(json.dumps({k: d[k] for k in ("groups", "layout", "assays")}))
    reproposed = False
    roles_changed = False

    if step == "layout":
        lay = decision.get("layout")
        if lay not in VOCABULARY["layout"]:
            raise StepError("Choose one of the three layouts.")
        reproposed = E("set_layout", {"layout": lay}, extra={"on_progress": on_progress})["result"]["reproposed"]
        d = s.draft
    if "assays" in decision:
        E("set_assays", {"assays": decision["assays"]})
    for ed in decision.get("items") or []:
        r = E("edit_group", {"group_id": ed.get("group_id"), "fields": {k: v for k, v in ed.items() if k != "group_id"}})
        roles_changed |= r["result"]["role_changed"]
    if "feature_identity" in decision:
        r = E("set_feature_identity", {"group_ids": decision["feature_identity"].get("group_ids", [])})
        roles_changed |= r["result"]["role_changed"]
    if "sample_id_group" in decision:
        E("set_sample_id_column", {"group_id": decision["sample_id_group"]})
    for gid, rule in (decision.get("sample_rules") or {}).items():
        E("set_sample_rule", dict(rule, group_id=gid))
    if "design" in decision:
        E("set_design", decision["design"])
    if decision.get("samples"):
        E("set_samples", {"samples": decision["samples"]})
    if "processing_history" in decision:
        E("set_processing_history", decision["processing_history"])
    for x in decision.get("derived_annotations") or []:
        E("set_derived_annotation", x)
    if decision.get("join_key"):
        E("set_join_key", {"metadata_column": decision["join_key"]})
    if "metadata" in decision:
        E("set_metadata", decision["metadata"])

    d = s.draft
    _check_step(s, d, step)
    refresh_samples(s, d)
    d["steps"][step] = "confirmed"
    if roles_changed and not reproposed:
        idx = STEPS.index(step)
        for later in ("annotations", "values", "samples", "sample_info"):
            if STEPS.index(later) > idx and d["steps"][later] == "confirmed":
                d["steps"][later] = "pending"
    s.log("step_confirmed", {"step": step, "decision": decision, "reproposed": reproposed,
                             "changes": _diff(before, d)})
    return reproposed


def after_edit(s):
    """Called by the edit layer after every edit: derived state (samples, which steps apply)."""
    d = s.draft
    if not d:
        return
    refresh_samples(s, d)
    for st in STEPS:
        if d["steps"][st] == "pending" and not step_applicable(s, d, st):
            d["steps"][st] = "not_applicable"
        elif d["steps"][st] == "not_applicable" and step_applicable(s, d, st):
            d["steps"][st] = "pending"
    questions.refresh_code(s, d)


def code_question_candidates(s, d):
    """The situations code turns into questions (v2.4 §5). Each has a stable key, so an
    answered or dismissed one is never asked again."""
    out = []
    rep = consistency_report(s, d)
    for f in rep["fragmentation"]:
        applies = {"group_ids": sorted(f["group_ids"])}
        n, m = len(f["group_ids"]), len(f["assays"])
        same = ("have the same samples" if layout_of(d) == "samples_in_rows"
                else "hold different samples of the same features")
        edits_one = ([{"op": "merge_assays", "args": {"assay_labels": f["assays"], "label": f["assays"][0]}}] if m > 1 else []) + \
            [{"op": "merge_groups", "args": {"group_ids": f["group_ids"], "reason": "one measurement (your answer)"}}]
        out.append({"kind": "fragmentation", "type": "single", "key": questions.key_of("fragmentation", applies),
                    "applies_to": applies, "step": "values", "evidence": f["evidence"], "allow_free_text": True,
                    "text": f"{n} blocks in {m} assay{'s' if m > 1 else ''} {same} and near-identical value "
                            "distributions. One measurement or several?",
                    "options": [
                        {"label": f"One measurement: one {'assay and one ' if m > 1 else ''}block ({f['n_columns']} columns)",
                         "edits": edits_one, "n_columns": f["n_columns"]},
                        {"label": "Separate measurements: keep them as they are", "edits": []},
                        {"label": "Ask the AI", "chat": f"Are these {n} blocks one measurement or several? "
                                                        + "; ".join(f["evidence"][:6])}]})
    out += _orphan_questions(s, d)
    out += design_questions(s, d)
    for c in rep["sample_collisions"]:
        applies = {"group_ids": sorted(c["group_ids"])}
        labels = {g: re.sub(r"[^A-Za-z0-9]+", "_", (d["groups"][g].get("label") or g)).strip("_")[:20] or g
                  for g in c["group_ids"]}
        out.append({"kind": "sample_id_collision", "type": "single", "key": questions.key_of("sample_id_collision", applies),
                    "applies_to": applies, "step": "samples", "text": c["message"],
                    "evidence": [f"Colliding IDs: {', '.join(c['ids'][:8])}"], "allow_free_text": True, "options": [
                        {"label": "Use the full column names as sample IDs",
                         "edits": [{"op": "resolve_consistency", "args": {"action": "full_names", "group_ids": c["group_ids"]}}]},
                        {"label": "Put each block's label in front of its IDs (" + ", ".join(labels.values()) + ")",
                         "edits": [{"op": "resolve_consistency", "args": {"action": "labels", "group_ids": c["group_ids"],
                                                                          "labels": labels}}]}]})
    for x in d.get("derived_feature_annotations", []):
        if not x.get("keep", True):
            continue
        if x.get("n_empty") and "" not in x.get("display_labels", {}):
            applies = {"derived": x["name"], "value": ""}
            out.append({"kind": "unclassified_part", "type": "single", "key": questions.key_of("unclassified_part", applies),
                        "applies_to": applies, "step": "annotations",
                        "text": f"{x['n_empty']} feature name(s) have an empty {x['name'].replace('_', ' ')}. "
                                "Show these as 'unclassified'? (Only a display label; the stored value stays empty.)",
                        "allow_free_text": False, "options": [
                            {"label": "Yes, show them as 'unclassified'",
                             "edits": [{"op": "set_derived_annotation", "args": {"name": x["name"],
                                                                                 "display_labels": {"": "unclassified"}}}]},
                            {"label": "No, leave them as (empty)", "edits": []}]})
        if (x.get("coverage") or 0) < 1:
            applies = {"derived": x["name"], "coverage": x["coverage"]}
            out.append({"kind": "derivation_coverage", "type": "single", "key": questions.key_of("derivation_coverage", applies),
                        "applies_to": applies, "step": "annotations",
                        "text": f"The rule for '{x['name']}' did not apply to {x['n_failures']} feature name(s) "
                                f"(coverage {round(100 * x['coverage'])}%). Keep it (those features get an empty value)?",
                        "allow_free_text": True, "options": [
                            {"label": "Keep it", "edits": []},
                            {"label": "Remove the derived column",
                             "edits": [{"op": "set_derived_annotation", "args": {"name": x["name"], "keep": False}}]}]})
    return out


def _orphan_questions(s, d):
    """Columns the AI left unplaced (v2.4 §16): a question with up to 3 candidate blocks
    (same name template first, then the closest median), its own block, or exclude."""
    from .profiling import name_template
    if not d["ai"].get("used"):
        return []
    blocks = [g for g in value_blocks(s, d) if s.groups_by_id[g]["type"] == "numeric"]
    tpl = {b: {name_template(c) for c in s.groups_by_id[b]["columns"]} for b in blocks}
    out = []
    orphans = [g for g in s.groups if d["groups"][g["group_id"]]["role"] == UNRESOLVED
               and g["origin"] in ("unmentioned", "unconsolidated")]
    for g in orphans[:40]:
        gid = g["group_id"]
        applies = {"columns": g["columns"][:50]}
        mine = {name_template(c) for c in g["columns"]}
        med = (g.get("profile") or {}).get("median")
        def rank(b):
            bm = (s.groups_by_id[b].get("profile") or {}).get("median")
            dist = abs(math.log10(med / bm)) if med and bm and med > 0 and bm > 0 else 99
            return (0 if mine & tpl[b] else 1, dist)
        opts = []
        if g["type"] == "numeric":
            for b in sorted(blocks, key=rank)[:3]:
                it = d["groups"][b]
                opts.append({"label": f"Add to '{it.get('label') or s.groups_by_id[b]['columns'][0]}' "
                                      f"({s.groups_by_id[b]['n_columns']} columns, {it.get('assay_label')})",
                             "edits": [{"op": "merge_groups", "args": {"group_ids": [b, gid],
                                                                       "reason": "your answer: same block"}}]})
            opts.append({"label": "Keep it as its own block", "edits": [{"op": "edit_group", "args": {
                "group_id": gid, "fields": {"role": "value"}}}]})
        else:
            role = "sample_metadata" if layout_of(d) == "samples_in_rows" else "feature_annotation"
            opts.append({"label": "It describes the " + ("samples" if role == "sample_metadata" else "features"),
                         "edits": [{"op": "edit_group", "args": {"group_id": gid, "fields": dict(
                             {"role": role}, **({"audit_kind": "other"} if role == "sample_metadata" else {}))}}]})
        opts.append({"label": "Exclude it from the outputs", "edits": [{"op": "edit_group", "args": {
            "group_id": gid, "fields": {"role": "ignore", "keep": False}}}]})
        name = ", ".join(g["columns"][:2]) + (" …" if g["n_columns"] > 2 else "")
        out.append({"kind": "orphan", "type": "single", "key": questions.key_of("orphan", applies), "applies_to": applies,
                    "step": step_for_group(s, d, gid), "allow_free_text": True, "options": opts,
                    "text": f"The AI did not place {name}. Where does it belong?",
                    "evidence": [group_hint(g)]})
    return out


def answer_question(s, qid, option_ids, text=None):
    questions.answer(s, qid, option_ids, text)
    return {"draft": public_draft(s)}


def dismiss_question(s, qid, note=None):
    questions.dismiss(s, qid, note)
    return {"draft": public_draft(s)}


def _invariants(s, d):
    """Problems that no edit may introduce (v2.4 §4.4), keyed so only new ones count."""
    out = {}
    name = lambda g: (s.groups_by_id.get(g) or {"columns": [g]})["columns"][0]
    for g in d["feature_identity"]["group_ids"]:
        it = d["groups"].get(g)
        if it and (it["role"] == "ignore" or not it.get("keep", True)):
            out[("feature_id", g)] = f"The feature ID column '{name(g)}' cannot be excluded."
    sg = (d.get("sample_id_group") or {}).get("value")
    it = d["groups"].get(sg) if sg else None
    if it and (it["role"] == "ignore" or not it.get("keep", True)):
        out[("sample_id", sg)] = f"The sample ID column '{name(sg)}' cannot be excluded."
    if not any(it["role"] == "value" and it.get("keep", True) for it in d["groups"].values()):
        out[("value",)] = "At least one value block must stay kept."
    return out


def check_invariants(s, before, tx):
    """Called by the edit layer before an edit is committed: a violation rejects the edit."""
    if before is None or s.draft is None:
        return
    new = _invariants(s, s.draft)
    old = _invariants(s, before)
    probs = [m for k, m in new.items() if k not in old and not (k == ("value",) and tx["actor"] == "user")]
    if tx["actor"] != "user" and before.get("processing_history") != s.draft.get("processing_history"):
        probs.append("Processing history is answered by you only; an AI patch cannot change it.")
    if probs:
        raise StepError(" ".join(probs))


# ---------------------------------------------------------------- ops (wizard controls; see edits.py)

@edits.op("set_layout")
def _op_set_layout(ctx, a):
    s, d = ctx.s, ctx.d
    lay = a.get("layout")
    if lay not in VOCABULARY["layout"]:
        raise StepError("Choose one of the three layouts.")
    if lay == d["layout"]["value"]:
        return {"reproposed": False}
    kept = {k: d.get(k) for k in ("processing_history", "software_and_version", "history_notes")}
    history_confirmed = d["steps"].get("history") == "confirmed"
    build_draft(s, ai_on=d["ai"]["enabled"], fixed_layout=lay, on_progress=ctx.extra.get("on_progress"))
    nd = s.draft
    nd.update({k: v for k, v in kept.items() if v is not None})
    for st in STEPS[1:]:
        nd["steps"][st] = "pending"
    if history_confirmed:
        nd["steps"]["history"] = "confirmed"
    nd["layout"]["set_by"] = ctx.actor
    ctx.summary = f"Layout: {lay.replace('_', ' ')} (everything re-proposed)"
    return {"reproposed": True}


@edits.op("set_assays")
def _op_set_assays(ctx, a):
    d = ctx.d
    new = a.get("assays") or []
    if not new:
        raise StepError("Describe at least one assay.")
    labels = [_clean(x.get("assay_label"), 120) for x in new]
    if not all(labels):
        raise StepError("Every assay needs a label.")
    for k, x in enumerate(new):
        label = labels[k]
        if k < len(d["assays"]):
            old = d["assays"][k]
            if old["assay_label"] != label:
                for it in d["groups"].values():
                    if it.get("assay_label") == old["assay_label"]:
                        it["assay_label"] = label
                    if (it.get("proposed") or {}).get("assay_label") == old["assay_label"]:
                        it["proposed"]["assay_label"] = label  # a rename is not a correction of the block
        else:
            old = _snap({"source": "user", "confidence": 1.0, "evidence": "Added by you."}, ASSAY_FIELDS)
            d["assays"].append(old)
        before = {f: old.get(f) for f in ASSAY_FIELDS}
        old["assay_label"] = label
        old["omics_type"] = _clean(x.get("omics_type"), 80) or "unknown"
        if x.get("omics_family") is not None:
            if x["omics_family"] not in VOCABULARY["omics_family"]:
                raise StepError(f"omics family must be one of: {', '.join(VOCABULARY['omics_family'])}.")
            old["omics_family"] = x["omics_family"]
        old["source_software"] = _clean(x.get("source_software"), 120) or "unknown"
        if x.get("in_supported_scope") in VOCABULARY["in_supported_scope"]:
            old["in_supported_scope"] = x["in_supported_scope"]
        if "scope_reason" in x:
            old["scope_reason"] = _clean(x.get("scope_reason"))
        if any(old.get(f) != before[f] for f in ASSAY_FIELDS):
            old["set_by"] = ctx.actor
    del d["assays"][len(new):]
    ctx.summary = "Assays: " + ", ".join(labels)


@edits.op("set_feature_identity")
def _op_set_feature_identity(ctx, a):
    s, d = ctx.s, ctx.d
    gids = [g for g in dict.fromkeys(a.get("group_ids") or []) if g in d["groups"]]
    changed = False
    for gid, it in list(d["groups"].items()):
        if gid in gids and it["role"] != "feature_id":
            changed |= edits.set_group_fields(ctx, gid, {"role": "feature_id"})
        elif gid not in gids and it["role"] == "feature_id":
            changed |= edits.set_group_fields(ctx, gid, {"role": "feature_annotation"})
    fi = d["feature_identity"]
    if fi["group_ids"] != gids:
        fi["set_by"] = ctx.actor
    fi["group_ids"], fi["composite"] = gids, len(gids) > 1
    fi["validation"] = validate_feature_identity(fi, s.groups_by_id, s.cols, layout_of(d))
    ctx.summary = "Feature ID: " + (" + ".join(s.groups_by_id[g]["columns"][0] for g in gids) or "none")
    return {"role_changed": changed}


@edits.op("set_sample_id_column")
def _op_set_sample_id_column(ctx, a):
    s, d = ctx.s, ctx.d
    gid = a.get("group_id")
    if gid is not None and gid not in d["groups"]:
        raise StepError(f"Unknown group '{gid}'.")
    for g2, it in list(d["groups"].items()):
        if it["role"] == "sample_id" and g2 != gid:
            edits.set_group_fields(ctx, g2, {"role": "sample_metadata", "audit_kind": "other"})
    if gid:
        edits.set_group_fields(ctx, gid, {"role": "sample_id"})
    if d["sample_id_group"]["value"] != gid:
        d["sample_id_group"]["set_by"] = ctx.actor
    d["sample_id_group"]["value"] = gid
    ctx.summary = "Sample ID column: " + (s.groups_by_id[gid]["columns"][0] if gid else "none")


@edits.op("set_sample_rule")
def _op_set_sample_rule(ctx, a):
    d = ctx.d
    gid = a.get("group_id")
    rule = d["sample_rules"].get(gid)
    if rule is None:
        return
    new = {"strip_prefix": a.get("strip_prefix", rule.get("strip_prefix", "")),
           "strip_suffix": a.get("strip_suffix", rule.get("strip_suffix", ""))}
    if "add_prefix" in a:
        new["add_prefix"] = _clean(a.get("add_prefix"), 60)
    if any(rule.get(k, "") != v for k, v in new.items()):
        rule.update(new)
        rule["set_by"] = ctx.actor
        ctx.summary = f"Sample IDs of {ctx.s.groups_by_id[gid]['columns'][0]} …"


def _set_samples(ctx, edits_by_sample):
    d = ctx.d
    n = 0
    for sid, ed in edits_by_sample.items():
        if sid not in d["samples"]:
            continue
        it = d["samples"][sid]
        before = (it.get("label"), it.get("is_study_sample"))
        if ed.get("label") is not None:
            it["label"] = _clean(ed["label"], 120)
        if ed.get("is_study_sample") is not None:
            it["is_study_sample"] = bool(ed["is_study_sample"])
        it["user_edited"] = True
        if (it.get("label"), it.get("is_study_sample")) != before:
            it["set_by"], it["set_corrected"] = ctx.actor, bool(ctx.corrected)
            n += 1
    ctx.summary = f"{n} sample(s) relabelled"
    return n


@edits.op("set_samples")
def _op_set_samples(ctx, a):
    return _set_samples(ctx, a.get("samples") or {})


@edits.op("set_sample_label", ai=True)
def _op_set_sample_label(ctx, a):
    ids = [x for x in a.get("sample_ids") or [] if x in ctx.d["samples"]]
    if not ids:
        raise StepError("No sample matches.")
    if a.get("label") is None and a.get("is_study_sample") is None:
        raise StepError("Nothing to change: give a label and/or is_study_sample.")
    return _set_samples(ctx, {x: {"label": a.get("label"), "is_study_sample": a.get("is_study_sample")} for x in ids})


@edits.op("set_processing_history")
def _op_set_processing_history(ctx, ph):
    """Answered by you only (v2.4 §10): an AI patch cannot reach this op."""
    d = ctx.d
    for q, _ in HISTORY_QUESTIONS:
        a = ph.get(q) or {}
        if a.get("answer") not in VOCABULARY["yes_no_unsure"]:
            raise StepError("Answer every processing-history question (yes, no or not sure).")
    for q, _ in HISTORY_QUESTIONS:
        a, cur = ph[q], d["processing_history"].get(q) or {}
        new = dict(cur, answer=a["answer"], note=_clean(a.get("note")), provenance="user_set")
        if cur.get("answer") != a["answer"] or not cur.get("answered_at"):
            new["answered_at"] = edits.now_iso()
        d["processing_history"][q] = new
    d["software_and_version"] = _clean(ph.get("software_and_version"))
    d["history_notes"] = _clean(ph.get("notes"), 2000)
    ctx.summary = "Processing history answered"


@edits.op("merge_assays", ai=True)
def _op_merge_assays(ctx, a):
    """Blocks of the merged assays become blocks of one assay (v2.4 §16)."""
    d = ctx.d
    labels = list(dict.fromkeys(a.get("assay_labels") or []))
    known = [x["assay_label"] for x in d["assays"]]
    bad = [x for x in labels if x not in known]
    if bad:
        raise StepError(f"Unknown assay(s): {', '.join(bad)}.")
    if len(labels) < 2:
        raise StepError("merge_assays needs at least two assays.")
    keep = next(x for x in d["assays"] if x["assay_label"] in labels)
    target = _clean(a.get("label"), 120) or keep["assay_label"]
    if target in known and target not in labels:
        raise StepError(f"'{target}' is already another assay.")
    old_label = keep["assay_label"]
    keep["assay_label"], keep["set_by"] = target, ctx.actor
    d["assays"] = [x for x in d["assays"] if x is keep or x["assay_label"] not in labels]
    for gid, it in list(d["groups"].items()):
        if it.get("assay_label") in labels or it.get("assay_label") == old_label:
            if it["role"] == "value":
                edits.set_group_fields(ctx, gid, {"assay_label": target})
            else:
                it["assay_label"] = target
    ctx.summary = f"Merged assays {', '.join(labels)} into '{target}'"


@edits.op("set_join_key", ai=True)
def _op_set_join_key(ctx, a):
    s, d = ctx.s, ctx.d
    meta = d.get("metadata")
    if not meta or meta.get("skipped") or not s.metadata_table:
        raise StepError("Upload a metadata file first.")
    col = a.get("metadata_column")
    by = {c["column"]: c for c in meta["columns"]}
    if col not in by:
        raise StepError(f"Metadata column '{col}' does not exist.")
    new = by[col]
    if new["role"] == "sample_id":
        return
    for c in meta["columns"]:
        if c["role"] == "sample_id":
            c.update(role="sample_metadata", audit_kind=c.get("audit_kind") or "other", label=c.get("label") or c["column"])
            c["set_by"] = ctx.actor
    new.update(role="sample_id", keep=True, set_by=ctx.actor)
    new.pop("excluded", None)
    meta["id_column"] = col
    t = s.metadata_table["table"]
    meta["report"] = match_report(sample_ids(s, d), [cell(r, new["index"]).strip() for r in t["rows"]])
    meta["accepted_near_misses"] = []
    ctx.columns.append(("metadata", col))
    ctx.summary = f"Metadata joined on '{col}' ({meta['report']['n_matched']} samples matched)"


def feature_names(s, d, source, column=None):
    """The names a feature-name rule applies to: value-column headers (samples in rows)
    or the values of an identifier column (samples in columns / long)."""
    lay = layout_of(d)
    if source == "column_headers":
        if lay != "samples_in_rows":
            raise StepError("Feature names are column headers only when samples are in rows.")
        return [c for gid in value_blocks(s, d) for c in s.groups_by_id[gid]["columns"]]
    if source == "feature_id_column":
        gid = None
        if column:
            gid = next((g["group_id"] for g in s.groups if column in g["columns"]), None)
            if gid is None:
                raise StepError(f"Column '{column}' does not exist.")
        elif d["feature_identity"]["group_ids"]:
            gid = d["feature_identity"]["group_ids"][0]
        if gid is None or d["groups"][gid]["role"] not in ("feature_id", "feature_annotation"):
            raise StepError("Choose the feature ID (or annotation) column whose values are split.")
        i = s.groups_by_id[gid]["indices"][0]
        return [cell(r, i).strip() for r in s.table["rows"]]
    raise StepError("source must be 'column_headers' or 'feature_id_column'.")


def derivation_preview(s, d, a):
    rule = a.get("rule") or {}
    taken = set(s.cols.labels) | {x["name"] for x in d.get("derived_feature_annotations", [])}
    try:
        names = derive.check_parts(rule, taken)
        out = derive.preview(feature_names(s, d, a.get("source"), a.get("column")), rule, names)
    except derive.RuleError as e:
        raise StepError(str(e))
    return names, out


@edits.op("derive_feature_annotation", ai=True)
def _op_derive_feature_annotation(ctx, a):
    """Adds derived annotation columns from feature names; never renames or changes anything."""
    s, d = ctx.s, ctx.d
    names, prev = derivation_preview(s, d, a)
    if prev["n_parsed"] == 0:
        raise StepError("The delimiter occurs in none of the feature names.")
    rule = {k: a["rule"][k] for k in ("delimiter", "occurrence")}
    parts = a["rule"]["parts"]
    for k, n in enumerate(names):
        if not n:
            continue
        d.setdefault("derived_feature_annotations", []).append({
            "name": n, "label": _clean(parts[k].get("label")) or n.replace("_", " "), "part": ("left", "right")[k],
            "derived_from": "feature_names", "source": a.get("source"), "column": a.get("column"), "rule": rule,
            "coverage": prev["coverage"], "n_failures": prev["n_failures"], "display_labels": {},
            "n_distinct": prev["parts"][n]["n_distinct"], "n_empty": prev["parts"][n]["n_empty"],
            "keep": True, "set_by": ctx.actor, "set_corrected": bool(ctx.corrected)})
    ctx.summary = (f"Derived {', '.join(n for n in names if n)} from feature names "
                   f"('{rule['delimiter']}', {rule['occurrence']}; {prev['n_parsed']}/{prev['n_total']} parsed)")
    return prev


@edits.op("set_derived_annotation")
def _op_set_derived_annotation(ctx, a):
    """Your edits of a derived annotation: label, keep, or a display label for a value
    (e.g. show the empty class as 'unclassified'; the stored value stays empty)."""
    der = {x["name"]: x for x in ctx.d.get("derived_feature_annotations", [])}
    x = der.get(a.get("name"))
    if x is None:
        raise StepError(f"No derived annotation '{a.get('name')}'.")
    if "label" in a:
        x["label"] = _clean(a["label"]) or x["label"]
    if "keep" in a:
        x["keep"] = bool(a["keep"])
    for v, lab in (a.get("display_labels") or {}).items():
        if lab:
            x["display_labels"][v] = _clean(lab, 60)
        else:
            x["display_labels"].pop(v, None)
    x["set_by"] = ctx.actor
    ctx.summary = f"Derived annotation '{x['name']}' updated"


@edits.op("set_metadata")
def _op_set_metadata(ctx, md):
    apply_metadata_decision(ctx.s, ctx.d, md, ctx)
    ctx.summary = "Sample metadata file: " + ("skipped" if md.get("skip") else "columns confirmed")


def _check_step(s, d, step):
    lay = layout_of(d)
    if step == "feature_id" and lay in ("samples_in_columns", "long") and not d["feature_identity"]["group_ids"]:
        raise StepError("Choose the column(s) that identify each feature.")
    if step == "feature_id" and lay == "long":
        if not d["sample_id_group"]["value"]:
            raise StepError("Choose the column that names the sample on each row.")
        d["long_duplicates"] = long_duplicates(s, d)
    if step == "values" and not value_blocks(s, d):
        raise StepError("Keep at least one value block.")
    if step == "samples" and lay in ("samples_in_rows", "long") and not d["sample_id_group"]["value"]:
        raise StepError("Choose the column that identifies each sample.")


def _diff(before, after):
    out = []
    for gid, it in after["groups"].items():
        b = before["groups"].get(gid)
        if b is None:
            continue
        ch = {f: [b.get(f), it.get(f)] for f in GROUP_FIELDS if b.get(f) != it.get(f)}
        if ch:
            out.append({"group_id": gid, "changes": ch, "proposal": it.get("proposed"),
                        "provenance": provenance(it, GROUP_FIELDS)})
    if before["layout"]["value"] != after["layout"]["value"]:
        out.append({"field": "layout", "from": before["layout"]["value"], "to": after["layout"]["value"],
                    "provenance": provenance(after["layout"], FACT_FIELDS)})
    if [a.get("assay_label") for a in before["assays"]] != [a.get("assay_label") for a in after["assays"]] or \
            any(b.get(f) != a.get(f) for b, a in zip(before["assays"], after["assays"]) for f in ASSAY_FIELDS):
        out.append({"field": "assays", "from": [{f: a.get(f) for f in ASSAY_FIELDS} for a in before["assays"]],
                    "to": [{f: a.get(f) for f in ASSAY_FIELDS} for a in after["assays"]]})
    return out


def long_duplicates(s, d):
    fid = [i for gid in d["feature_identity"]["group_ids"] for i in s.groups_by_id[gid]["indices"]]
    sg = d["sample_id_group"]["value"]
    if not fid or not sg:
        return None
    si = s.groups_by_id[sg]["indices"][0]
    c = Counter(("|".join(cell(r, i).strip() for i in fid), cell(r, si).strip()) for r in s.table["rows"])
    dups = [(k, n) for k, n in c.items() if n > 1]
    if not dups:
        return None
    return {"n_pairs": len(dups), "examples": [{"feature": k[0], "sample": k[1], "rows": n} for k, n in dups[:5]],
            "message": ("Multiple rows per feature and sample: aggregation is required and is not supported "
                        "in this version. Nothing was aggregated.")}


# ---------------------------------------------------------------- reconsider (one group or a whole step)

def _replace_groups(s, d, old_gids, pgroups, why, ctx=None):
    """Replace groups old_gids by the proposed pgroups (same columns, regrouped).
    A proposed group with exactly the columns of an old one keeps its id (and the
    wizard keeps its place); new structures get new ids and their steps reopen."""
    old = {g: s.groups_by_id[g] for g in old_gids}
    by_cols = {frozenset(g["indices"]): gid for gid, g in old.items()}
    keep_ids = {pid: by_cols[frozenset(pg["indices"])] for pid, pg in pgroups.items()
                if frozenset(pg["indices"]) in by_cols}
    keep_ids.update({pid: pg["keep_id"] for pid, pg in pgroups.items() if pg.get("keep_id") in old})
    layout = layout_of(d)
    old_steps = {step_for_group(s, d, g) for g in old_gids if g in d["groups"]}
    new_groups, items, id_map = install_groups(s, pgroups, layout, keep_ids, first_id=_next_id(s))
    s.groups = sorted([g for g in s.groups if g["group_id"] not in old] + new_groups, key=lambda g: min(g["indices"]))
    labels = [a["assay_label"] for a in d["assays"]]
    structural = any(g["group_id"] not in old or frozenset(g["indices"]) != frozenset(old[g["group_id"]]["indices"])
                     for g in new_groups) or len(new_groups) != len(old)
    reopen = old_steps if structural else set()
    for g in new_groups:
        gid, it = g["group_id"], items[g["group_id"]]
        prev = d["groups"].get(gid)
        it["hint"] = group_hint(g)
        if prev is not None:
            it["keep"] = prev.get("keep", True)   # your keep / exclude choice survives a re-proposal
        if it["role"] == "value" and it.get("assay_label") not in labels:
            it["assay_label"] = (prev or {}).get("assay_label") if (prev or {}).get("assay_label") in labels else labels[0]
        _attach_flags(s, gid, it)
        if "proposed" not in it:
            _snap(it, GROUP_FIELDS)
        d["groups"][gid] = it
        if g.get("sample_id_rule") is not None and gid not in d["sample_rules"]:
            d["sample_rules"][gid] = _snap(dict(g["sample_id_rule"], source="computed"), ("strip_prefix", "strip_suffix"))
        if structural:
            reopen.add(step_for_group(s, d, gid))
    gone = [g for g in old if g not in {x["group_id"] for x in new_groups}]
    for g in gone:
        d["groups"].pop(g, None)
        d["sample_rules"].pop(g, None)
    for g in new_groups:  # a group whose columns changed needs a fresh sample rule
        if g["group_id"] in old and structural and g.get("sample_id_rule") is not None:
            d["sample_rules"][g["group_id"]] = _snap(dict(g["sample_id_rule"], source="computed"),
                                                     ("strip_prefix", "strip_suffix"))
    d["groups"] = {g["group_id"]: d["groups"][g["group_id"]] for g in s.groups}
    fi = d["feature_identity"]
    fi["group_ids"] = [g for g in fi["group_ids"] if g in d["groups"]]
    if d["sample_id_group"]["value"] not in d["groups"]:
        d["sample_id_group"]["value"] = None
    for st in reopen:
        if st in d["steps"] and d["steps"][st] == "confirmed":
            d["steps"][st] = "pending"
    refresh_samples(s, d)
    if ctx is not None:
        ctx.columns.extend(("main", c) for g in old.values() for c in g["columns"])
    s.log("regroup", {"why": why, "old": {g: old[g]["columns"] for g in old},
                      "new": {g["group_id"]: g["columns"] for g in new_groups}, "reopened_steps": sorted(reopen)})
    return [g["group_id"] for g in new_groups]


def _check_gids(d, gids, at_least=1):
    gids = [g for g in dict.fromkeys(gids if isinstance(gids, (list, tuple)) else [gids]) if g]
    unknown = [g for g in gids if g not in d["groups"]]
    if len(gids) < at_least or unknown:
        raise StepError(f"Unknown group(s): {', '.join(unknown) or 'none given'}." if unknown or not gids else
                        f"Choose at least {at_least} groups.")
    return gids


def reconsider(s, gids, hint, on_progress=None):
    """Re-ask the AI about these groups' columns with the user's note. It may
    relabel them, regroup them, split columns out or merge them; code applies the
    answer only where the column names and ids exist."""
    d = s.draft
    gids = _check_gids(d, gids)
    ok, why = llm.available()
    if not (ok and d["ai"]["enabled"]):
        raise StepError("The AI is off or unavailable: " + (why or "turn it on to ask it to reconsider."))
    others = [(g, it) for g, it in d["groups"].items() if g not in gids and it["role"] != UNRESOLVED]
    fixed = {"layout": d["layout"]["value"],
             "assays": [{f: a.get(f) for f in ASSAY_FIELDS} for a in d["assays"]],
             "current_grouping_of_these_columns": [
                 {"group_id": g, "columns": s.groups_by_id[g]["columns"],
                  **{k: d["groups"][g].get(k) for k in ("role", "label", "audit_kind")}} for g in gids],
             "confirmed_groups_elsewhere": {g: {k: it.get(k) for k in ("role", "label", "audit_kind", "keep")}
                                            for g, it in others[:200]},
             "user_feedback": _clean(hint, 1000),
             "instruction": ("The user disagrees with the previous proposal for these columns. Reconsider only "
                             "them, taking the user's feedback into account: relabel, regroup, split or merge.")}
    cols = sorted(i for g in gids for i in s.groups_by_id[g]["indices"])
    prop, meta, digests = ai.propose(s.filename, s.sha, s.cols, s.affixes, s.hints, indices=cols, fixed=fixed,
                                     log=s.log, on_progress=on_progress, signature_hint=d.get("signature_hint"))
    if meta.get("all_failed") or (meta.get("error") and not any(pg["origin"] != "ai_unavailable"
                                                                 for pg in prop["groups"].values())):
        raise StepError("The AI could not answer: " + (meta.get("error") or "no answer"))
    old_cols = {g: set(s.groups_by_id[g]["indices"]) for g in gids}
    new_ids = edits.apply_edit(s, "regroup", {"group_ids": gids, "why": {"reconsider": hint}},
                               extra={"pgroups": prop["groups"]},
                               summary=f"The AI reconsidered {len(gids)} group(s): {_clean(hint, 80)}")["result"]
    d = s.draft
    split_ids = [g for g in new_ids if g not in old_cols and s.groups_by_id[g].get("split_from")]
    s.log("reconsider", {"group_ids": gids, "user_feedback": hint, "result": new_ids,
                         "rejected": prop.get("rejected"), "merges": prop.get("merges_applied"),
                         "splits": prop.get("splits_applied")})
    s.save()
    return {"draft": public_draft(s), "digest": digests[0] if digests else None, "split_groups": split_ids,
            "new_groups": new_ids, "questions": [q["question_id"] for q in (
                questions.from_ai(s, d, x, "reconsider") for x in prop.get("clarifying_questions", [])) if q]}


def retry_consolidation(s):
    """Run the final-answer call again on the current groups (e.g. after a quota error):
    groups it joins are merged; labels, roles and assays come from its answer."""
    d = s.draft
    ok, why = llm.available()
    if not (ok and d["ai"]["enabled"]):
        raise StepError("The AI is off or unavailable: " + (why or ""))
    resp, err = ai.consolidate(s.cols, s.groups_by_id, d["groups"], d["assays"], s.sha, log=s.log)
    if err:
        raise StepError("The AI could not answer: " + err)
    applied, rejected = edits.apply_edit(s, "apply_consolidation", {}, extra={"resp": resp},
                                         summary="The AI's final answer for the whole file")["result"]
    s.log("consolidation_retry", {"applied": applied, "rejected": rejected})
    s.save()
    return {"draft": public_draft(s), "merges": applied}


@edits.op("apply_consolidation")
def _op_apply_consolidation(ctx, a):
    """A new proposal (asked for by you): merges, final labels and assays from the AI's answer."""
    s, d = ctx.s, ctx.d
    resp = ctx.extra["resp"]
    applied, rejected, claimed = [], [], set()
    finals = []
    for fg in resp.groups:
        gids = [g for g in dict.fromkeys(fg.members) if g in d["groups"] and g not in claimed]
        bad = [g for g in fg.members if g not in d["groups"]]
        if bad:
            rejected.append({"group_id": fg.group_id, "members": bad, "reason": "names groups that do not exist"})
        if not gids:
            continue
        claimed.update(gids)
        if len(gids) > 1:
            keep = max(gids, key=lambda g: s.groups_by_id[g]["n_columns"])
            pg = {"m": {"indices": [i for g in gids for i in s.groups_by_id[g]["indices"]],
                        "item": copy.deepcopy(d["groups"][keep]), "origin": "merged_consolidation",
                        "merged_from": gids, "keep_id": keep}}
            _replace_groups(s, d, gids, pg, {"consolidation": gids}, ctx)
            applied.append({"group_ids": gids, "into": keep, "reason": fg.evidence})
            gids = [keep]
        finals.append((gids[0], fg))
    if resp.assays:
        old = d["assays"]
        d["assays"] = [_snap(dict(ai._assay_entry(x)), ASSAY_FIELDS) for x in resp.assays]
        for x in d["assays"]:
            x.pop("feature_pids", None)
    labels = [x["assay_label"] for x in d["assays"]]
    for gid, fg in finals:
        it = d["groups"][gid]
        new = ai.finalize_item(_blank(ai._final_item(fg)), s.groups_by_id[gid], s.cols, layout_of(d))
        if new["role"] == "value" and new.get("assay_label") not in labels:
            new["assay_label"] = labels[0]
        new["hint"] = it.get("hint")
        new["keep"] = it.get("keep", True)
        _attach_flags(s, gid, new)
        d["groups"][gid] = _snap(new, GROUP_FIELDS)
    for gid, it in d["groups"].items():
        if it["role"] == "value" and it.get("assay_label") not in labels:
            it["assay_label"] = labels[0]
    d["grouping"]["consolidation_error"] = None
    d["grouping"]["merges_applied"] = d["grouping"].get("merges_applied", []) + applied
    d["rejected"] = d.get("rejected", []) + rejected
    ctx.summary = f"Final answer: {len(finals)} group(s) relabelled, {len(applied)} merge(s)"
    return applied, rejected


@edits.op("regroup")
def _op_regroup(ctx, a):
    """Apply an AI regrouping of these groups' columns (asked for by you)."""
    return _replace_groups(ctx.s, ctx.d, a["group_ids"], ctx.extra["pgroups"], a.get("why"), ctx)


def merge_groups(s, gids, why=None):
    """Your decision: merge these groups into one."""
    new = edits.apply_edit(s, "merge_groups", {"group_ids": gids, "reason": why})["result"]
    s.save()
    return {"draft": public_draft(s), "new_groups": new}


@edits.op("merge_groups", ai=True)
def _op_merge_groups(ctx, a):
    """The group with the most columns keeps its id and description; the result is
    re-validated (e.g. text + numbers cannot be a value block)."""
    s, d = ctx.s, ctx.d
    gids, why = _check_gids(d, a.get("group_ids") or [], 2), a.get("reason")
    keep = max(gids, key=lambda g: s.groups_by_id[g]["n_columns"])
    item = copy.deepcopy(d["groups"][keep])
    who = "you" if ctx.actor == "user" else "an AI patch you applied"
    item.update(source="user" if ctx.actor == "user" else item.get("source"), set_by=ctx.actor,
                evidence=f"Merged by {who}: {', '.join(gids)}." + (f" {why}" if why else ""))
    ctx.summary = f"Merged {len(gids)} groups ({sum(s.groups_by_id[g]['n_columns'] for g in gids)} columns)"
    for g in gids:
        ctx.touch(g)
    pg = {"m": {"indices": [i for g in gids for i in s.groups_by_id[g]["indices"]], "item": item,
                "origin": "merged_user", "merged_from": gids, "keep_id": keep}}
    return _replace_groups(s, d, gids, pg, {"merge": gids, "reason": why}, ctx)


def split_columns(s, gid, columns):
    """Your decision: take these columns out of the group, one group each."""
    new = edits.apply_edit(s, "split_group", {"group_id": gid, "columns": columns})["result"]
    s.save()
    return {"draft": public_draft(s), "new_groups": new}


@edits.op("split_group", ai=True)
def _op_split_group(ctx, a):
    s, d = ctx.s, ctx.d
    gid, columns = a.get("group_id"), a.get("columns") or []
    _check_gids(d, [gid])
    g = s.groups_by_id[gid]
    take = [g["indices"][g["columns"].index(c)] for c in columns if c in g["columns"]]
    if not take:
        raise StepError("Choose at least one column of this group.")
    if len(take) == g["n_columns"]:
        raise StepError("That is every column of the group; nothing to take out.")
    pg = {"r": {"indices": [i for i in g["indices"] if i not in take], "item": copy.deepcopy(d["groups"][gid]),
                "origin": g["origin"], "keep_id": gid}}
    for i in take:
        pg[f"c{i}"] = {"indices": [i], "origin": "split_user", "split_from": gid,
                       "item": dict(grouping.unmentioned_item(), source="user",
                                    evidence=f"Taken out of {gid}: choose what it is.")}
    ctx.touch(gid)
    ctx.summary = f"Took {len(take)} column(s) out of {g['columns'][0]} …"
    return _replace_groups(s, d, [gid], pg, {"split": gid, "columns": columns}, ctx)


def group_columns(s, columns, why=None):
    """Your decision: these columns are one group (e.g. all columns sharing a name part)."""
    new = edits.apply_edit(s, "group_columns", {"columns": columns, "reason": why})["result"]
    s.save()
    return {"draft": public_draft(s), "new_groups": new}


@edits.op("group_columns")
def _op_group_columns(ctx, a):
    """They leave their current groups; what is left of those groups stays as it was."""
    s, d = ctx.s, ctx.d
    columns, why = a.get("columns") or [], a.get("reason")
    idx = [s.cols.labels.index(c) for c in dict.fromkeys(columns) if c in s.cols.labels]
    if len(idx) < 2:
        raise StepError("Choose at least two columns.")
    owner = {i: g["group_id"] for g in s.groups for i in g["indices"]}
    touched = list(dict.fromkeys(owner[i] for i in idx))
    donor = max(touched, key=lambda g: len(set(s.groups_by_id[g]["indices"]) & set(idx)))
    item = copy.deepcopy(d["groups"][donor])
    item.update(source="user", evidence="Grouped by you" + (f": {why}" if why else "."))
    pg = {"new": {"indices": idx, "item": item, "origin": "grouped_user"}}
    for g in touched:
        rest = [i for i in s.groups_by_id[g]["indices"] if i not in idx]
        if rest:
            pg[f"r{g}"] = {"indices": rest, "item": copy.deepcopy(d["groups"][g]), "origin": s.groups_by_id[g]["origin"],
                           "keep_id": g}
    if not any(pg[p].get("keep_id") == donor for p in pg):
        pg["new"]["keep_id"] = donor
    for g in touched:
        ctx.touch(g)
    ctx.summary = f"Grouped {len(idx)} columns" + (f" ({why})" if why else "")
    return _replace_groups(s, d, touched, pg, {"group_columns": len(idx), "reason": why}, ctx)


def name_parts(s, limit=15):
    """Literal name parts shared by several columns (facts, for grouping by hand)."""
    seen = getattr(s, "_name_parts", None)
    if seen is None:
        seen = s._name_parts = _name_part_index(s)
    owner = {i: g["group_id"] for g in s.groups for i in g["indices"]}
    label_idx = {c: i for i, c in enumerate(s.cols.labels)}
    out = []
    for v in seen.values():
        groups = {owner.get(label_idx[c]) for c in v["columns"]}
        if len(groups) > 1:  # only parts that are not already one group
            out.append(dict(v, n=len(v["columns"]), n_groups=len(groups)))
    # display order only: most shared text first (columns x characters); nothing is hidden or grouped
    out.sort(key=lambda v: (-v["n"] * len(v["text"].strip()), -v["n"]))
    return out[:limit]


def _with_part(s, txt, prefix):
    """Column labels starting (or ending) with txt, by binary search on sorted names."""
    import bisect
    if not hasattr(s, "_sorted_names"):
        s._sorted_names = (sorted(s.cols.labels), sorted(c[::-1] for c in s.cols.labels))
    srt = s._sorted_names[0 if prefix else 1]
    key = txt if prefix else txt[::-1]
    hit = srt[bisect.bisect_left(srt, key):bisect.bisect_left(srt, key + "\U0010ffff")]
    return hit if prefix else [c[::-1] for c in hit]


def _name_part_index(s):
    seen = {}
    for i, a in enumerate(s.affixes):
        for side in ("shared_prefix", "shared_suffix"):
            for x in a.get(side) or []:
                if not x["text"].strip():
                    continue
                key = (side, x["text"])
                if key not in seen:
                    txt = x["text"]
                    cols = _with_part(s, txt, side == "shared_prefix")
                    seen[key] = {"side": "prefix" if side == "shared_prefix" else "suffix", "text": txt, "columns": cols}
    return seen


# Literature retrieval (backend/literature.py) is deferred to a future Tier 2 feature, not
# abandoned: Step 0 runs before the research-focus intake, so it cannot yet say which block
# should be analysed. The module (with its quote verification) is kept but not called here.


# ---------------------------------------------------------------- talking to the AI: chat with patches (v2.4 §4)

def _cols_brief(g):
    c = g["columns"]
    return c if len(c) <= 100 else c[:3] + [f"... ({len(c) - 6} more)"] + c[-3:]


def chat_context(s, d, message, step=None, selection=None):
    """What the AI receives each turn (v2.4 §4.6): a compact draft summary, the digest facts
    of the columns referred to, the last 10 chat turns, the step and selection. Never raw rows."""
    groups = []
    for g in s.groups:
        it = d["groups"][g["group_id"]]
        x = {"group_id": g["group_id"], "role": it["role"], "label": it.get("label") or "", "n_columns": g["n_columns"],
             "columns": _cols_brief(g), "keep": it.get("keep", True)}
        for k in ("audit_kind", "assay_label", "family"):
            if it.get(k):
                x[k] = it[k]
        if it.get("marks_rows_as_suspect"):
            x["flagged_values"] = it.get("flagged_values")
        groups.append(x)
    columns = [{"column": c, "file": "main", "group_id": g["group_id"], "role": d["groups"][g["group_id"]]["role"],
                "label": d["groups"][g["group_id"]].get("label") or "",
                "audit_kind": d["groups"][g["group_id"]].get("audit_kind"),
                "keep": d["groups"][g["group_id"]].get("keep", True)}
               for g in s.groups if d["groups"][g["group_id"]]["role"] not in ("value",) and g["kind"] != "numeric_block"
               for c in g["columns"]]
    meta = d.get("metadata") or {}
    if not meta.get("skipped"):
        columns += [{"column": c["column"], "file": "metadata", "role": c["role"], "label": c.get("label") or "",
                     "audit_kind": c.get("audit_kind"), "keep": c.get("keep", True)} for c in meta.get("columns", [])]
    low = (message or "").lower()
    refer = [c for c in (selection or []) if c in s.cols.labels]
    refer += [c for c in s.cols.labels if len(c) > 2 and c.lower() in low and c not in refer]
    idx = {c: i for i, c in enumerate(s.cols.labels)}
    facts = [ai.column_digest(s.cols, idx[c], s.affixes[idx[c]], ai.send_examples()) for c in refer[:40]]
    ids = list(d["samples"])
    labels = Counter(v.get("label") for v in d["samples"].values())
    return {"message": message, "step": step, "selection": selection or [],
            "layout": d["layout"]["value"],
            "assays": [{"assay_label": a["assay_label"], "omics_type": a.get("omics_type"),
                        "n_value_blocks": sum(1 for it in d["groups"].values()
                                              if it["role"] == "value" and it.get("assay_label") == a["assay_label"])}
                       for a in d["assays"]],
            "groups": groups, "columns": columns,
            "feature_id": [s.groups_by_id[g]["columns"][0] for g in d["feature_identity"]["group_ids"]],
            "sample_id_column": (s.groups_by_id[d["sample_id_group"]["value"]]["columns"][0]
                                 if d["sample_id_group"]["value"] else None),
            "samples": {"n": len(ids), "first": ids[:50], "labels": dict(labels.most_common(10))},
            "design": d.get("design"),
            "open_questions": [{"question_id": q["question_id"], "text": q["text"]}
                               for q in d.get("questions", []) if q.get("status") == "open"],
            "excluded_columns": [x["column"] for x in accounting.excluded_columns(s, d)][:300],
            "derived_feature_annotations": [x["name"] for x in d.get("derived_feature_annotations", [])],
            "digest_of_referred_columns": facts,
            "history": [{"user": m["text"], "assistant": m.get("reply", "")} for m in d.get("chat", [])[-10:]],
            "settings": {"raw_rows_sent": False}}


def chat(s, message, step=None, selection=None):
    """Send your message to the AI. Its reply is stored with the patches it proposes
    (checked and previewed by code, applied only when you click)."""
    from . import patches as P
    d = s.draft
    message = _clean(message, 2000)
    if not message:
        raise StepError("Write a message first.")
    ok, why = llm.available()
    if not (ok and d["ai"]["enabled"]):
        raise StepError("The AI is off or unavailable: " + (why or "turn it on to talk to it.") +
                        " Manual editing and the questions still work.")
    payload = chat_context(s, d, message, step, selection)
    resp, err = ai.chat(payload, s.sha, log=s.log)
    if resp is None:
        raise StepError("The AI could not answer: " + (err or "no answer"))
    mid = uuid.uuid4().hex[:8]
    made = [P.prepare(s, p.model_dump(), "chat", mid) for p in resp.patches]
    qs = [q for q in (questions.from_ai(s, d, q.model_dump(), "chat") for q in resp.questions) if q]
    msg = {"message_id": mid, "at": edits.now_iso(), "text": message, "step": step, "selection": selection or [],
           "reply": resp.reply, "patch_ids": [p["patch_id"] for p in made],
           "question_ids": [q["question_id"] for q in qs], "context": payload}
    auto = []
    if (d.get("settings") or {}).get("auto_apply"):
        for p in made:   # what you asked for, unless it is large, warned about or near an invariant
            if p["status"] == "pending" and auto_applicable(s, d, p):
                r = P.apply(s, [p["patch_id"]], actor="ai_patch")[0]
                if r["status"] == "applied":
                    auto.append(p["patch_id"])
        d = s.draft
    msg["auto_applied"] = auto
    d.setdefault("chat", []).append(msg)
    del d["chat"][:-50]
    s.log("chat_message", {"message_id": mid, "text": message, "step": step, "selection": selection or [],
                           "reply": resp.reply, "patches": msg["patch_ids"],
                           "rejected": [p["patch_id"] for p in made if p["status"] == "rejected"]})
    s.save()
    return {"draft": public_draft(s), "message": {k: v for k, v in msg.items() if k != "context"}}


def auto_applicable(s, d, p):
    """Auto-apply (an opt-in per session) never covers ops over 25 columns, patches with
    warnings, or ops near an invariant (excluding a value block, merging, assays, join key)."""
    if p.get("large") or p.get("warnings"):
        return False
    if p["op"] in ("set_block_keep", "merge_groups", "split_group", "merge_assays", "set_join_key", "set_role"):
        return False
    if p["op"] == "set_keep" and (p.get("args") or {}).get("keep") is False:
        cols = set((p.get("resolved") or {}).get("columns") or [])
        if any(it["role"] in ("value", "feature_id", "sample_id") for g, it in d["groups"].items()
               if set(s.groups_by_id[g]["columns"]) & cols):
            return False
    return True


def set_settings(s, auto_apply=None):
    d = s.draft
    st = d.setdefault("settings", {})
    if auto_apply is not None:
        st["auto_apply"] = bool(auto_apply)
        s.log("settings", {"auto_apply": st["auto_apply"]})
    s.save()
    return {"draft": public_draft(s)}


def apply_patches(s, patch_ids, confirm_large=(), overrides=None):
    from . import patches as P
    res = P.apply(s, patch_ids, confirm_large, overrides)
    return {"draft": public_draft(s), "results": res}


def dismiss_patches(s, patch_ids):
    from . import patches as P
    P.dismiss(s, patch_ids)
    return {"draft": public_draft(s)}


# ---------------------------------------------------------------- consistency (v2.3): flags you confirm

def consistency_report(s, d):
    """Deterministic checks: near-identical identifier-named blocks, and sample-ID
    collisions between blocks of one assay (samples in columns)."""
    lay = layout_of(d)
    frag = consistency.fragmentation(s.groups_by_id, d["groups"], lay,
                                     {g: sample_ids(s, d, g) for g in value_blocks(s, d)} if lay == "samples_in_columns" else {})
    collisions, structure = [], {}
    if layout_of(d) == "samples_in_columns":
        by_assay = {}
        for gid in value_blocks(s, d):
            by_assay.setdefault(d["groups"][gid].get("assay_label"), []).append(gid)
        for assay, gids in by_assay.items():
            ids = {g: sample_ids(s, d, g) for g in gids}
            codes = {g: consistency.name_code(s.groups_by_id[g]) for g in gids}
            found, structure[assay] = consistency.sample_collisions(ids, codes)
            for c in found:
                first = c["ids"][0]
                collisions.append(dict(c, kind="sample_id_collision", assay=assay,
                                       message=(f"Sample ID '{first}' appears in more than one block"
                                                + (f" ({c['n_ids']} IDs in total)" if c["n_ids"] > 1 else "")
                                                + " — these need to be distinguished.")))
    return {"fragmentation": frag, "sample_collisions": collisions, "sample_structure": structure}


@edits.op("resolve_consistency")
def _op_resolve_consistency(ctx, a):
    """Telling colliding sample IDs apart (v2.3 §3): full column names, or a label per block."""
    s, d = ctx.s, ctx.d
    action, group_ids, labels = a["action"], a.get("group_ids"), a.get("labels")
    if action not in ("full_names", "labels"):
        raise StepError(f"Unknown action '{action}'.")
    ctx.summary = {"full_names": "Full column names as sample IDs", "labels": "Block labels added to sample IDs"}[action]
    gids = [g for g in (group_ids or []) if g in d["sample_rules"]]
    if not gids:
        raise StepError("Choose the blocks whose sample IDs collide.")
    for g in gids:
        rule = d["sample_rules"][g]
        if action == "full_names":
            rule.update(strip_prefix="", strip_suffix="", add_prefix="")
        else:
            lab = _clean((labels or {}).get(g), 60)
            if not lab:
                raise StepError("Give every block a short label.")
            rule["add_prefix"] = lab if lab.endswith(("_", "-", ".", " ")) else lab + "_"
        rule["set_by"] = ctx.actor
    s.log("consistency", {"action": action, "group_ids": gids, "labels": labels})
    if d["steps"].get("samples") == "confirmed":
        d["steps"]["samples"] = "pending"


# ---------------------------------------------------------------- study design (v2.4 §7)

DESIGN_FIELDS = ("source", "column", "file")


def design_column_values(s, d, file, column):
    """{sample: value} of a sample-level column, from the main file (samples in rows) or the metadata file."""
    if file == "metadata":
        return design.column_values_meta(s, d, column)
    if layout_of(d) == "samples_in_columns":
        return {}
    g = next((g for g in s.groups if column in g["columns"]), None)
    if g is None:
        return {}
    i = g["indices"][g["columns"].index(column)]
    ids = sample_ids(s, d)
    if layout_of(d) == "long":
        return {}
    return {sid: ("" if is_missing(cell(r, i)) else cell(r, i).strip()) for sid, r in zip(ids, s.table["rows"])}


def design_candidates(s, d):
    """Sample-level columns a subject / time may come from: (file, column, audit_kind)."""
    out = []
    if layout_of(d) == "samples_in_rows":
        for g in s.groups:
            it = d["groups"][g["group_id"]]
            if it["role"] == "sample_metadata" and g["n_columns"] == 1:
                out.append(("main", g["columns"][0], it.get("audit_kind")))
    meta = d.get("metadata") or {}
    if not meta.get("skipped"):
        for c in meta.get("columns", []):
            if c["role"] == "sample_metadata":
                out.append(("metadata", c["column"], c.get("audit_kind")))
    return out


def design_values(s, d):
    """{sample: {"subject": .., "time": ..}} from the chosen sources, plus the derivation result."""
    des = d.get("design") or design.blank()
    ids = sample_ids(s, d)
    per = {sid: {} for sid in ids}
    der = None
    rule = des.get("derivation")
    if rule and "derived_from_sample_names" in (des["subject"]["source"], des["time"]["source"]):
        try:
            vals, fails = design.derived_values(ids, rule)
            der = {"n_parsed": len(vals), "n_total": len(ids), "failures": fails[:30], "n_failures": len(fails),
                   "coverage": round(len(vals) / len(ids), 4) if ids else 0.0}
            for side in ("subject", "time"):
                if des[side]["source"] == "derived_from_sample_names":
                    for sid in ids:
                        per[sid][side] = vals.get(sid, {}).get(side)
        except derive.RuleError as e:
            der = {"error": str(e)}
    for side in ("subject", "time"):
        src = des[side]
        if src["source"] == "metadata_column" and src.get("column"):
            vals = design_column_values(s, d, src.get("file"), src["column"])
            for sid in ids:
                per[sid][side] = vals.get(sid)
    return per, der


def design_report(s, d):
    """Computed design facts for the Design step and schema.json (v2.4 §7.1)."""
    des = d.get("design") or design.blank()
    per, der = design_values(s, d)
    facts = design.summarize(per)
    facts["derivation"] = der
    facts["coverage"] = {side: sum(1 for v in per.values() if v.get(side) not in (None, "")) for side in ("subject", "time")}
    facts["n_samples"] = len(per)
    cands = design_candidates(s, d)
    facts["candidates"] = [{"file": f, "column": c, "audit_kind": k} for f, c, k in cands]
    subj = {sid: v.get("subject") for sid, v in per.items()}
    varies = []
    if des["subject"]["source"] != "none":
        for f, c, k in cands:
            if (f, c) == (des["subject"].get("file"), des["subject"].get("column")):
                continue
            varies.append({"file": f, "column": c, "audit_kind": k,
                           "varies_within_subject": design.varies_within_subject(per, design_column_values(s, d, f, c))})
    facts["varies_within_subject"] = varies
    checks = []
    sides = {"subject": ("subject_id",), "time": ("timepoint",)}
    for side, kinds in sides.items():
        chosen = {sid: v.get(side) for sid, v in per.items()}
        if des[side]["source"] == "none":
            continue
        name0 = ("derived_" + side) if des[side]["source"] == "derived_from_sample_names" else f"{side}:{des[side]['column']}"
        others = [(f"{f}:{c}", design_column_values(s, d, f, c)) for f, c, k in cands
                  if k in kinds and (f, c) != (des[side].get("file"), des[side].get("column"))]
        rule = des.get("derivation")
        if des[side]["source"] == "metadata_column" and rule and side in (rule.get("left"), rule.get("right")):
            vals, _ = design.derived_values(list(per), rule)
            others.append((f"derived_{side}", {sid: v.get(side) for sid, v in vals.items()}))
        for name, vals in others:
            checks.append(design.cross_check(f"{name0}_vs_{name}", chosen, vals))
    facts["cross_checks"] = checks
    facts["subject_values"] = subj if len(subj) <= 2000 else None
    return facts


def preview_derivation(s, rule):
    """Live preview of a sample-name rule (sample, subject, time), with coverage and failures. Nothing is set."""
    d = s.draft
    rule = {k: (rule or {}).get(k) for k in ("delimiter", "occurrence", "left", "right")}
    try:
        out = design.preview(sample_ids(s, d), rule)
    except derive.RuleError as e:
        raise StepError(str(e))
    vals, _ = design.derived_values(sample_ids(s, d), rule)
    out["summary"] = design.summarize(vals)
    out["candidates"] = design.derivation_candidates(sample_ids(s, d))
    s.log("design_derivation_previewed", {"rule": rule, "coverage": out["coverage"], "n_failures": out["n_failures"]})
    return out


def _validate_design_source(s, d, side, src):
    if src.get("source") not in design.SOURCES:
        raise StepError(f"{side} source must be one of {', '.join(design.SOURCES)}.")
    if src["source"] == "metadata_column":
        cands = {(f, c) for f, c, _ in design_candidates(s, d)}
        f = src.get("file") or ("metadata" if any(x[0] == "metadata" and x[1] == src.get("column") for x in cands) else "main")
        if (f, src.get("column")) not in cands:
            raise StepError(f"'{src.get('column')}' is not a sample information column (of the {f} file).")
        src["file"] = f
    else:
        src["column"], src["file"] = None, None
    return src


@edits.op("set_design", ai=True)
def _op_set_design(ctx, a):
    """Where subject and time come from (partial update). The time unit is your answer only."""
    s, d = ctx.s, ctx.d
    des = d.setdefault("design", _snap_design(design.blank(), "none"))
    new = json.loads(json.dumps(des))
    if "derivation" in a and a["derivation"] is not None:
        rule = {k: a["derivation"].get(k) for k in ("delimiter", "occurrence", "left", "right")}
        try:
            design.check_derivation(rule)
        except derive.RuleError as e:
            raise StepError(str(e))
        new["derivation"] = rule
    for side in ("subject", "time"):
        x = a.get(side)
        if not x:
            continue
        if "unit" in x:
            if ctx.actor == "ai_patch":
                raise StepError("The time unit is asked, never inferred: you answer it in the Design step.")
            if x["unit"] not in design.UNITS:
                raise StepError(f"Unit must be one of: {', '.join(design.UNITS)}.")
            new["time"]["unit"] = {"value": x["unit"], "provenance": "user_set", "answered_at": edits.now_iso()}
        if "source" in x:
            src = _validate_design_source(s, d, side, {k: x.get(k) for k in DESIGN_FIELDS})
            new[side].update(src)
            if src["source"] == "derived_from_sample_names" and not new.get("derivation"):
                raise StepError("Give the rule that splits the sample names (delimiter, first / last, which side is what).")
    if new != des:
        sources_changed = False
        for side in ("subject", "time"):
            if {k: new[side].get(k) for k in DESIGN_FIELDS} != {k: des[side].get(k) for k in DESIGN_FIELDS} \
                    or (new.get("derivation") != des.get("derivation") and new[side]["source"] == "derived_from_sample_names"):
                new[side]["set_by"], new[side]["set_corrected"] = ctx.actor, bool(ctx.corrected)
                sources_changed = True
        d["design"] = new
        if sources_changed and d["steps"].get("design") == "confirmed" and ctx.step != "design":
            d["steps"]["design"] = "pending"
    ctx.summary = "Design: subject " + _src_text(new["subject"], new) + ", time " + _src_text(new["time"], new)


def _src_text(src, des):
    if src["source"] == "metadata_column":
        return f"from '{src['column']}'"
    if src["source"] == "derived_from_sample_names":
        r = des.get("derivation") or {}
        return f"from the sample names ('{r.get('delimiter')}', {r.get('occurrence')})"
    return "none"


def _snap_design(des, source="ai"):
    for side in ("subject", "time"):
        des[side].setdefault("confidence", 0.0)
        des[side]["source_of_proposal"] = source
        des[side]["proposed"] = {k: des[side].get(k) for k in DESIGN_FIELDS}
        des[side]["proposed_derivation"] = des.get("derivation")
    return des


def design_provenance(des, side):
    x = des[side]
    changed = {k: x.get(k) for k in DESIGN_FIELDS} != (x.get("proposed") or {}) or (
        x.get("source") == "derived_from_sample_names" and des.get("derivation") != x.get("proposed_derivation"))
    if not changed:
        return {"ai": "ai_proposed_confirmed", "computed": "computed"}.get(x.get("source_of_proposal"), "user_set")
    if x.get("set_by") in ("ai_patch", "question_option"):
        return "ai_proposed_corrected" if x.get("set_corrected") else "ai_proposed_confirmed"
    return "user_set"


def design_from_ai(s, d, raw, origin):
    """An AI design proposal (main proposal or metadata call), validated; invalid parts are dropped."""
    if not raw:
        return None
    des = design.blank()
    rule = raw.get("derivation")
    if rule:
        try:
            r = {k: rule.get(k) for k in ("delimiter", "occurrence", "left", "right")}
            design.check_derivation(r)
            des["derivation"] = r
        except derive.RuleError:
            pass
    for side in ("subject", "time"):
        x = raw.get(side) or {}
        src = {k: x.get(k) for k in DESIGN_FIELDS}
        if src.get("source") == "derived_from_sample_names" and not des["derivation"]:
            continue
        try:
            des[side].update(_validate_design_source(s, d, side, src) if src.get("source") else {"source": "none"})
        except StepError:
            continue
    des["evidence"] = raw.get("evidence") or ""
    des["confidence"] = raw.get("confidence") or 0.0
    des["origin"] = origin
    return _snap_design(des, "ai")


def design_questions(s, d):
    des = d.get("design")
    out = []
    if not des:
        return out
    rep = design_report(s, d)
    t = des["time"]
    if t["source"] != "none" and not (t.get("unit") or {}).get("value"):
        applies = {"time": {k: t.get(k) for k in DESIGN_FIELDS}, "derivation": des.get("derivation")
                   if t["source"] == "derived_from_sample_names" else None}
        vals = rep.get("time_values") or []
        hint = ""
        name = (t.get("column") or "") + " " + " ".join(map(str, vals[:12]))
        if re.search(r"(?i)\bweek|\bwk|^w\d|\sw\d", name):
            hint = " The names or values mention weeks; that is only a hint."
        elif re.search(r"(?i)\bday|\bd\d", name):
            hint = " The names or values mention days; that is only a hint."
        where = f"'{t['column']}'" if t["source"] == "metadata_column" else "the part of the sample names"
        out.append({"kind": "time_unit", "type": "single", "key": questions.key_of("time_unit", applies),
                    "applies_to": applies, "step": "design", "allow_free_text": False,
                    "text": f"What unit is the time in {where} (values {', '.join(map(str, vals[:8]))}"
                            f"{' …' if len(vals) > 8 else ''})?{hint}",
                    "options": [{"label": u, "edits": [{"op": "set_design", "args": {"time": {"unit": u}}}]}
                                for u in design.UNITS]})
    der = rep.get("derivation") or {}
    if der.get("n_failures") and der.get("n_parsed"):
        applies = {"derivation": des.get("derivation")}
        sides = [x for x in ("subject", "time") if des[x]["source"] == "derived_from_sample_names"]
        out.append({"kind": "design_derivation_coverage", "type": "single", "step": "design", "allow_free_text": True,
                    "key": questions.key_of("design_derivation_coverage", applies), "applies_to": applies,
                    "text": f"The sample-name rule parsed {der['n_parsed']} of {der['n_total']} sample names "
                            f"(not: {', '.join(der['failures'][:5])}{' …' if der['n_failures'] > 5 else ''}). Use it anyway?",
                    "options": [{"label": f"Use it: those {der['n_failures']} sample(s) get no {' / '.join(sides)}", "edits": []},
                                {"label": "Do not derive: choose another source",
                                 "edits": [{"op": "set_design", "args": {x: {"source": "none"} for x in sides}}]}]})
    return out


# ---------------------------------------------------------------- sample metadata file

def _norm_id(x):
    x = re.sub(r"[\s_\-.]+", "", x.strip().lower())
    return re.sub(r"(?<![0-9])0+(?=[0-9])", "", x)


def upload_metadata(s, filename, raw):
    t = parse_bytes(sanitize_filename(filename), raw)   # a parse error leaves the session as it was
    with edits.recording(s, "upload_metadata", "user", f"Metadata file {sanitize_filename(filename)}"):
        _upload_metadata(s, filename, t)
    s.save()
    return public_draft(s)


def metadata_join_facts(cols, t, data_ids):
    """Per metadata column: how many of the data's sample names its values contain, exactly and
    after normalising case / spaces / separators / leading zeros (v2.4 §6). Facts, not a choice."""
    ds = set(data_ids)
    norm = {_norm_id(x) for x in data_ids}
    out = []
    for i in range(len(t["header"])):
        vals = {cell(r, i).strip() for r in t["rows"]} - {""}
        ex = len(vals & ds)
        nm = len({_norm_id(v) for v in vals} & norm)
        out.append({"column": cols.labels[i], "exact_matches": ex, "normalized_matches": nm,
                    "exact_share": round(ex / len(ds), 4) if ds else 0.0,
                    "normalized_share": round(nm / len(ds), 4) if ds else 0.0})
    return out


def _metadata_data_ctx(s, d, ids):
    lay = layout_of(d)
    from .profiling import value_shapes
    return {"layout": lay, "n_samples": len(ids),
            "sample_source": "value column headers" if lay == "samples_in_columns" else "sample ID column",
            "sample_name_shapes": value_shapes(Counter(ids)),
            "derivation_candidates": design.derivation_candidates(ids),
            "current_design": {k: (d.get("design") or {}).get(k) for k in ("subject", "time", "derivation")}}


def _upload_metadata(s, filename, t):
    """The metadata file's columns go through the AI like any other column (v2.4 §6): role,
    audit_kind, a specific label, confidence and evidence, plus the join key it proposes from the
    join facts. Without the AI: the join key with the most exact matches is suggested (computed)
    and every audit kind is left for you to choose. Nothing is guessed from column names by code."""
    cols = Columns(t)
    d = s.draft
    ids = sample_ids(s, d)
    facts = metadata_join_facts(cols, t, ids)
    s.metadata_table = {"table": t, "cols": cols}
    columns = []
    for i in range(len(t["header"])):
        dg = cols.digests[i]
        columns.append({"column": cols.labels[i], "index": i, "role": "sample_metadata", "audit_kind": None,
                        "label": "", "detail": None, "family": None, "keep": True, "source": "none", "confidence": 0.0,
                        "evidence": "", "join": facts[i],
                        "hint": group_hint({"kind": "single_column", "type": dg["type"], "profile": dg, "n_columns": 1})})
    by = {c["column"]: c for c in columns}
    ranked = sorted(range(len(columns)), key=lambda i: (-facts[i]["exact_matches"], -facts[i]["normalized_matches"], i))
    key, key_src, key_ev, key_conf = columns[ranked[0]]["column"], "computed", "", 0.0
    f0 = facts[ranked[0]]
    key_ev = (f"{f0['exact_matches']} of {len(ids)} sample names found exactly in this column"
              + (f" ({f0['normalized_matches']} after normalising)" if f0["normalized_matches"] != f0["exact_matches"] else ""))
    ai_info = {"used": False, "error": None}
    answer = None
    ok, why = llm.available()
    if ok and d["ai"]["enabled"]:
        answer, meta_, digests = ai.propose_metadata(sanitize_filename(filename), s.sha, cols, facts,
                                                     _metadata_data_ctx(s, d, ids), log=s.log)
        ai_info = {"used": answer is not None, "error": meta_.get("error"), "model": meta_.get("model")}
        s.metadata_digests = digests
    if answer:
        jk = answer.get("join_key") or {}
        if jk.get("column") in by:
            key, key_src = jk["column"], "ai"
            key_ev, key_conf = jk.get("evidence") or key_ev, ai._clamp(jk.get("confidence"))
        for name, x in answer["columns"].items():
            c = by[name]
            role = x["role"] if x["role"] in ("sample_metadata", "ignore") else "sample_metadata"
            kind = x.get("audit_kind") if x.get("audit_kind") in VOCABULARY["audit_kind"] else None
            c.update(role=role, audit_kind=kind if role == "sample_metadata" else None, label=(x.get("label") or "").strip(),
                     family=(x.get("family") or "").strip() or None, detail=(x.get("detail") or "").strip() or None,
                     confidence=ai._clamp(x.get("confidence")), evidence=x.get("evidence") or "", source="ai")
            if role == "ignore":
                c["keep"] = False
                c["excluded"] = {"reason": "proposed as 'ignore' by the AI", "by": "user", "at": None}
            if kind == "timepoint" and not c["detail"]:
                c["detail"] = timepoint_detail(cols.digests[c["index"]])
    k = by[key]
    k.update(role="sample_id", audit_kind=None, keep=True, source=key_src if key_src == "ai" else k["source"],
             evidence=key_ev if key_src == "computed" or not k["evidence"] else k["evidence"])
    k.pop("excluded", None)
    for c in columns:
        _snap(c, ("role", "audit_kind", "label", "keep"))
    meta = {"filename": sanitize_filename(filename), "id_column": key, "join_key_source": key_src,
            "join_key_evidence": key_ev, "join_key_confidence": key_conf, "columns": columns,
            "accepted_near_misses": [], "skipped": False, "sha256": t["sha256"], "n_rows": len(t["rows"]),
            "n_columns": len(t["header"]), "parse_report": t["parse_report"], "ai": ai_info,
            "join_facts_top": [facts[i] for i in ranked[:5]],
            "sample_source": "value column headers" if layout_of(d) == "samples_in_columns" else "sample ID column",
            "report": match_report(ids, [cell(r, by[key]["index"]).strip() for r in t["rows"]])}
    d["metadata"] = meta
    if answer:
        if answer.get("design") and all(d["design"][x]["source"] == "none" for x in ("subject", "time")):
            des = design_from_ai(s, d, answer["design"], "metadata")
            if des:
                d["design"] = des
        for q in answer.get("questions", []):
            questions.from_ai(s, d, q, "metadata")
    s.log("metadata_upload", {"filename": meta["filename"], "id_column": key, "join_key_source": key_src,
                              "ai": ai_info, "join_facts_top": meta["join_facts_top"],
                              "report": {k: v for k, v in meta["report"].items() if k != "matched"}})


def match_report(data_ids, meta_ids):
    ds, ms = set(data_ids), set(i for i in meta_ids if i)
    only_data = [i for i in data_ids if i not in ms]
    only_meta = [i for i in meta_ids if i and i not in ds]
    norm_meta = {}
    for m in only_meta:
        norm_meta.setdefault(_norm_id(m), []).append(m)
    near = [{"data_id": dd, "metadata_id": m, "reason": "differs only in case, spaces/underscores or leading zeros"}
            for dd in only_data for m in norm_meta.get(_norm_id(dd), [])]
    return {"matched": sorted(ds & ms), "n_matched": len(ds & ms), "only_in_data": only_data,
            "only_in_metadata": only_meta, "near_misses": near}


def apply_metadata_decision(s, d, md, ctx=None):
    if md.get("skip"):
        d["metadata"] = {"skipped": True}
        return
    meta = d.get("metadata")
    if not meta or meta.get("skipped"):
        raise StepError("Upload a metadata file or choose to skip.")
    by_col = {c["column"]: c for c in meta["columns"]}
    eds = [(by_col[ed["column"]], ed) for ed in md.get("columns", [])
           if ed.get("column") in by_col and by_col[ed["column"]]["role"] != "sample_id"]
    for c, ed in eds:   # validate everything first: a failing decision changes nothing
        if ed.get("keep", True) and c["role"] == "sample_metadata" and ed.get("audit_kind") not in VOCABULARY["audit_kind"]:
            raise StepError(f"Choose an audit kind for metadata column '{c['column']}'.")
    valid = {(n["data_id"], n["metadata_id"]) for n in meta["report"]["near_misses"]}
    meta["accepted_near_misses"] = [p for p in md.get("accept_near_misses", []) if tuple(p) in valid]
    actor = ctx.actor if ctx else "user"
    for c, ed in eds:
        before = {k: c.get(k) for k in ("audit_kind", "label", "detail", "keep")}
        c["audit_kind"] = ed.get("audit_kind") if ed.get("audit_kind") in VOCABULARY["audit_kind"] else c.get("audit_kind")
        c["label"] = _clean(ed.get("label", c["label"]))
        c["detail"] = _clean(ed.get("detail", c.get("detail"))) or None
        c["keep"] = bool(ed.get("keep", True))
        if {k: c.get(k) for k in before} != before:
            c["set_by"] = actor
            if ctx:
                ctx.columns.append(("metadata", c["column"]))
        if not c["keep"] and not c.get("excluded"):
            c["excluded"] = {"reason": (ctx.reason if ctx else None) or "left out of the outputs", "by": actor,
                             "at": edits.now_iso()}
        elif c["keep"]:
            c.pop("excluded", None)


# ---------------------------------------------------------------- public view

def public_draft(s):
    """Draft + derived facts for the frontend (provenance, sample ids, labels in use)."""
    d = s.draft
    out = copy.deepcopy(d)
    for gid, it in out["groups"].items():
        it["provenance"] = provenance(it, GROUP_FIELDS)
        g = s.groups_by_id[gid]
        if it.get("flag_values") is None and g["n_columns"] == 1 and it["role"] in ("feature_annotation", UNRESOLVED) \
                and ((g.get("profile") or {}).get("n_unique") or 99) <= FLAG_MAX_VALUES:
            it["value_counts"] = flag_values(s, gid)  # lets the UI offer 'marks rows as suspect' at once
        if it.get("marks_rows_as_suspect") and it.get("flag_values") and it.get("flagged_values"):
            it["flag_counts"] = {v: it["flag_values"].get(v, 0) for v in it["flagged_values"]}
            it["n_flagged"] = sum(it["flag_counts"].values())
    out["layout"]["provenance"] = provenance(out["layout"], FACT_FIELDS)
    for a in out["assays"]:
        a["provenance"] = provenance(a, ASSAY_FIELDS)
    out["feature_identity"]["provenance"] = provenance(out["feature_identity"], FI_FIELDS)
    ids = sample_ids(s, d)
    out["sample_list"] = {"ids": ids, "n": len(ids), "duplicates": [k for k, n in Counter(ids).items() if n > 1]}
    out["blocks_samples"] = {gid: sample_ids(s, d, gid) for gid, it in d["groups"].items()
                             if it["role"] == "value" and layout_of(d) == "samples_in_columns"}
    for st in out["samples"].values():
        st["provenance"] = provenance(st, SAMPLE_FIELDS)
    out["suggestions"] = {  # labels already used in this session (for the open text fields)
        "label": sorted({it["label"] for it in d["groups"].values() if it.get("label")}),
        "assay_label": [a["assay_label"] for a in d["assays"]],
        "omics_type": sorted({a["omics_type"] for a in d["assays"]} | {"proteomics", "metabolomics"}),
        "source_software": sorted({a["source_software"] for a in d["assays"]} - {"unknown"}),
        "sample_label": sorted({x["label"] for x in d["samples"].values() if x.get("label")}),
    }
    out["unresolved"] = unresolved_items(s, d)
    out["name_parts"] = name_parts(s)
    out["ai_ungrouped"] = [g["group_id"] for g in s.groups if g.get("origin") == "ai_unavailable"]
    out["consistency"] = consistency_report(s, d)
    out["changes"] = edits.changes(s)
    out["chat"] = [{k: v for k, v in m.items() if k != "context"} for m in d.get("chat", [])]
    out["last_chat_context"] = d["chat"][-1].get("context") if d.get("chat") else None
    out["questions"] = questions.public(d)
    out["design_report"] = design_report(s, d)
    out["feature_facts"] = {g: block_feature_facts(s, d, g) for g in value_blocks(s, d)}
    out["near_duplicates"] = near_duplicates(s, d)
    if (out.get("metadata") or {}).get("columns"):
        for c in out["metadata"]["columns"]:
            c["provenance"] = provenance(c, ("role", "audit_kind", "label", "keep"))
    for side in ("subject", "time"):
        out["design"][side]["provenance"] = design_provenance(d["design"], side)
    led = accounting.column_ledger(s, d)
    out["column_ledger"] = {f: dict(x, text=accounting.ledger_text(x)) for f, x in led.items()}
    out["excluded_columns"] = accounting.excluded_columns(s, d)
    return out


def block_feature_facts(s, d, gid):
    """v2.4 §18 facts of one value block (cached by its columns and the layout)."""
    from .profiling import feature_facts
    key = (tuple(s.groups_by_id[gid]["indices"]), layout_of(d))
    cache = s.__dict__.setdefault("_ff_cache", {})
    if key not in cache:
        cache[key] = feature_facts(s.cols, list(key[0]), key[1])
    return cache[key]


def near_duplicates(s, d):
    """v2.4 §8.3: near-duplicate pairs among the columns outside value blocks (a fact)."""
    from .profiling import near_duplicate_pairs
    idx = tuple(sorted(i for g in s.groups if d["groups"][g["group_id"]]["role"] != "value" for i in g["indices"]))
    cache = s.__dict__.setdefault("_nd_cache", {})
    if idx not in cache:
        pairs, skipped = near_duplicate_pairs(s.cols, idx)
        if skipped:
            s.log("near_duplicates_skipped", {"reason": skipped})
        cache[idx] = {"pairs": pairs, "skipped": skipped}
    return cache[idx]


def unresolved_items(s, d):
    out = []
    if d["layout"]["value"] not in VOCABULARY["layout"]:
        out.append({"step": "layout", "what": "Layout is not decided."})
    for gid, it in d["groups"].items():
        name = ", ".join(s.groups_by_id[gid]["columns"][:2]) + (" ..." if s.groups_by_id[gid]["n_columns"] > 2 else "")
        if it["role"] == UNRESOLVED:
            out.append({"step": step_for_group(s, d, gid), "group_id": gid, "what": f"Role of {name} is unresolved."})
        elif it["role"] == "sample_metadata" and it.get("audit_kind") not in VOCABULARY["audit_kind"]:
            out.append({"step": "sample_info", "group_id": gid, "what": f"Audit kind of '{name}' is not chosen."})
        elif it.get("marks_rows_as_suspect") and it.get("flag_values") and not it.get("flagged_values"):
            out.append({"step": "annotations", "group_id": gid,
                        "what": f"Choose which value of '{name}' means 'flagged'."})
    lay = layout_of(d)
    if lay in ("samples_in_columns", "long") and not d["feature_identity"]["group_ids"]:
        out.append({"step": "feature_id", "what": "Feature identity is not chosen."})
    if lay in ("samples_in_rows", "long") and not d["sample_id_group"]["value"]:
        out.append({"step": "samples", "what": "Sample ID column is not chosen."})
    if d.get("long_duplicates") and lay == "long":
        out.append({"step": "feature_id", "what": d["long_duplicates"]["message"]})
    if not value_blocks(s, d):
        out.append({"step": "values", "what": "No value block is kept."})
    for q in questions.open_questions(d):
        out.append({"step": q.get("step") or "review", "question_id": q["question_id"],
                    "what": f"Open question: {q['text']}"})
    meta = d.get("metadata") or {}
    for c in meta.get("columns", []) if not meta.get("skipped") else []:
        if c["role"] == "sample_metadata" and c.get("keep", True) and c.get("audit_kind") not in VOCABULARY["audit_kind"]:
            out.append({"step": "sample_info", "what": f"Audit kind of metadata column '{c['column']}' is not chosen."})
    for p in accounting.ledger_problems(accounting.column_ledger(s, d)):
        out.append({"step": "review", "what": f"Column accounting: {p}."})
    if any(not d["processing_history"][q]["answer"] for q, _ in HISTORY_QUESTIONS):
        out.append({"step": "history", "what": "Processing-history questions are not all answered."})
    for st in STEPS[:-1]:
        if d["steps"][st] == "pending":
            out.append({"step": st, "what": f"Step '{st.replace('_', ' ')}' is not confirmed yet."})
    return out


def step_for_group(s, d, gid):
    it, g, lay = d["groups"][gid], s.groups_by_id[gid], layout_of(d)
    role = it["role"]
    if role == "feature_id":
        return "feature_id"
    if role == "feature_annotation":
        return "annotations"
    if role == "value":
        return "values"
    if role == "sample_id":
        return "samples"
    if role == "sample_metadata":
        return "sample_info"
    if g["kind"] == "numeric_block":
        return "values"
    if lay == "samples_in_rows":
        return "values" if g["type"] == "numeric" else "sample_info"
    return "annotations"
