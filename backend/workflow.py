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
import os
import re
import threading
import uuid
from collections import Counter, OrderedDict
from pathlib import Path

from . import ai, literature, llm_providers as llm
from .format_detect import signature_hint, signature_prefill
from .mock_llm import expand_sample_rules
from .parsing import cell, is_missing, parse_bytes, sanitize_filename
from .profiling import Columns, apply_rule, build_groups, layout_hints
from .schema import HISTORY_QUESTIONS, UNRESOLVED, VOCABULARY
from .session_log import log_event
from .validation import timepoint_detail, validate_feature_identity, validate_group

SESSIONS_DIR = Path(os.environ.get("PRISM_SESSIONS_DIR", Path(__file__).parent / "sessions"))
STEPS = ["layout", "feature_id", "annotations", "values", "samples", "sample_info", "history", "review"]
GROUP_FIELDS = ("role", "assay_label", "label", "audit_kind", "marks_rows_as_suspect",
                "flagged_value", "detail", "keep")
ASSAY_FIELDS = ("assay_label", "omics_type", "source_software", "in_supported_scope", "scope_reason")
FACT_FIELDS = ("value",)
FI_FIELDS = ("group_ids", "composite")
SAMPLE_FIELDS = ("label", "is_study_sample")
FLAG_MAX_VALUES = 5
_NON_STUDY = re.compile(r"(?i)(^|[^a-z])(qc|pool|pooled|blank|buffer|calib\w*|std|standard)([^a-z]|$)")
_FLAG_WORDS = {"+", "x", "yes", "y", "true", "1", "flag", "flagged", "reverse", "rev", "con"}
_MAX_SESSIONS = 20
_sessions = OrderedDict()
_lock = threading.Lock()


class StepError(Exception):
    pass


# ---------------------------------------------------------------- provenance

def provenance(item, fields):
    src = item.get("source") or "none"
    proposed = item.get("proposed") or {}
    changed = any(item.get(f) != proposed.get(f) for f in fields if f in proposed or f in item)
    if src == "computed":
        return "user_set" if changed else "computed"
    if src == "ai":
        return "ai_proposed_corrected" if changed else "ai_proposed_confirmed"
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
        self.splits = []  # accepted split decisions, re-applied on reload
        self.groups = build_groups(self.cols)
        self.hints = layout_hints(self.cols, self.groups)
        self.draft = None
        self.digests = []
        self.metadata_table = None
        self.lock = threading.RLock()

    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "state.json").write_text(json.dumps({
            "sid": self.sid, "filename": self.filename, "splits": self.splits, "draft": self.draft,
            "digests": self.digests}, ensure_ascii=False), encoding="utf-8")

    def log(self, event, payload):
        log_event(self.sid, event, payload)

    @property
    def groups_by_id(self):
        return {g["group_id"]: g for g in self.groups}


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
    for sp in st.get("splits", []):
        _apply_split(s, sp)
    s.draft, s.digests = st.get("draft"), st.get("digests", [])
    md = d / "metadata_source.csv"
    if md.exists() and s.draft and s.draft.get("metadata") and not s.draft["metadata"].get("skipped"):
        t = parse_bytes(s.draft["metadata"]["filename"], md.read_bytes())
        s.metadata_table = {"table": t, "cols": Columns(t)}
    _remember(s)
    return s


# ---------------------------------------------------------------- splits (deterministic decision)

def _apply_split(s, sp):
    g = s.groups_by_id.get(sp["group_id"])
    if g is None or g["n_columns"] < 2:
        return False
    cols = [c for c in sp["columns"] if c in g["columns"]]
    if not cols or g["n_columns"] - len(cols) < 2:
        return False
    keep_idx = [i for i, c in zip(g["indices"], g["columns"]) if c not in cols]
    new = []
    for c in cols:
        i = g["indices"][g["columns"].index(c)]
        d = s.cols.digests[i]
        new.append({"group_id": f"{g['group_id']}s{len(new) + 1}", "columns": [c], "indices": [i],
                    "n_columns": 1, "kind": "single_column", "origin": "split_on_ai_suggestion",
                    "pattern": None, "type": d["type"], "profile": d, "split_from": g["group_id"],
                    "split_reasons": [sp.get("evidence") or "AI suggested it does not belong to the block"]})
    from .profiling import block_profile
    g["indices"] = keep_idx
    g["columns"] = [s.cols.labels[i] for i in keep_idx]
    g["n_columns"] = len(keep_idx)
    g["profile"], g["histogram"] = block_profile(s.cols, keep_idx)
    s.groups.extend(new)
    s.groups.sort(key=lambda x: min(x["indices"]))
    s.splits.append(sp)
    return [n["group_id"] for n in new]


def _apply_ai_splits(s, prop, layout):
    """Apply the AI's split suggestions (code decides: see _apply_split) and give
    each new single-column group the role the AI suggested for it."""
    new_all = []
    for sp in prop.get("splits", []):
        new_ids = _apply_split(s, sp)
        if new_ids:
            s.log("split_applied", {"split": sp, "new_groups": new_ids})
            for gid in new_ids:
                item = {"role": sp.get("role") or UNRESOLVED, "audit_kind": sp.get("audit_kind"),
                        "label": sp.get("label") or "", "confidence": 0.7, "source": "ai",
                        "evidence": f"Split out of the block on the AI's suggestion: {sp.get('evidence', '')}"}
                prop["groups"][gid] = ai.finalize_item(_blank(item), s.groups_by_id[gid], s.cols, layout)
            new_all.extend(new_ids)
    return new_all


# ---------------------------------------------------------------- hints

def group_hint(g):
    p = g.get("profile") or {}
    if g["kind"] == "numeric_block":
        pat = (g.get("pattern") or {}).get("text")
        base = f"{g['n_columns']} numeric columns" + (f" sharing '{pat.strip()}'" if pat else " with similar values")
        return base + f"; median {p.get('median')}, {int((p.get('frac_zero') or 0) * 100)}% zeros."
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
    if h["largest_numeric_block_columns"]:
        r = h["rows_to_block_columns_ratio"]
        if r is not None and r >= 1:
            return (f"{h['n_rows']} rows and {h['numeric_block_columns_total']} columns in numeric blocks: "
                    "features usually outnumber samples, so rows are probably features (samples in columns).")
        return (f"Only {h['n_rows']} rows but {h['numeric_block_columns_total']} columns in numeric blocks: "
                "rows are probably samples (samples in rows).")
    return "No large block of numeric columns was found."


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
            "marks_rows_as_suspect": False, "flagged_value": None, "detail": None, "keep": True,
            "confidence": 0.0, "evidence": "", "source": "none", "validation": {"status": "ok", "messages": []}}
    base.update({k: v for k, v in item.items() if v is not None or k not in base})
    return base


def build_draft(s, ai_on=True, fixed_layout=None, on_progress=None):
    """AI proposal (signature appended as a hint) or, when the AI is off or
    unavailable, the signature pre-fill / manual hints -> a fresh draft."""
    hint = signature_hint(s.table["header"])
    ok, why = llm.available()
    use_ai = ai_on and ok
    prop, meta, digests = None, {"error": None}, []
    if use_ai:
        fixed = {"layout": fixed_layout} if fixed_layout else {}
        prop, meta, digests = ai.propose(s.filename, s.sha, s.groups, s.hints, s.cols, fixed=fixed, log=s.log,
                                         on_progress=on_progress, signature_hint=hint)
        _apply_ai_splits(s, prop, fixed_layout)
        if meta.get("error") and not prop.get("layout"):
            prop = None  # the AI failed entirely: fall back to manual starting points
    manual = prop is None
    if manual:
        prop = signature_prefill(s.table["header"], s.groups) or {}
    s.digests = digests

    d = {"schema_version": "0.2.1", "groups": {}, "sample_rules": {}, "samples": {}, "sample_rules_ai": [],
         "processing_history": {q: {"answer": None, "note": ""} for q, _ in HISTORY_QUESTIONS},
         "software_and_version": "", "history_notes": "", "metadata": None,
         "steps": {st: "pending" for st in STEPS}, "signature_hint": hint}
    d["ai"] = {"enabled": bool(ai_on), "available": ok, "unavailable_reason": None if ok else why,
               "used": use_ai and not manual, "provider": meta.get("provider") or llm.provider_name(),
               "model": meta.get("model"), "models_used": meta.get("models_used", []),
               "prompt_version": ai.PROMPT_VERSION, "temperature": llm.TEMPERATURE, "error": meta.get("error"),
               "cached": meta.get("cached", False)}

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
        assays = [{"assay_label": "assay 1", "omics_type": "unknown", "source_software": "unknown",
                   "in_supported_scope": "unsure", "scope_reason": "not described yet", "confidence": 0.0,
                   "evidence": "", "source": "none"}]
    d["assays"] = [_snap(a, ASSAY_FIELDS) for a in assays]

    # groups
    labels = [a["assay_label"] for a in d["assays"]]
    for g in s.groups:
        gid = g["group_id"]
        if gid in prop.get("groups", {}):
            item = _blank(prop["groups"][gid])
        else:
            item = _blank({"evidence": ("AI is off: choose this role yourself." if not use_ai else
                                        "No proposal for this group."), "source": "none"})
        item["hint"] = group_hint(g)
        if item["role"] == "value" and item.get("assay_label") not in labels:
            item["assay_label"] = labels[0]
        _attach_flags(s, gid, item)
        d["groups"][gid] = item
    for gid, item in d["groups"].items():
        if item["role"] != UNRESOLVED:
            item["validation"] = validate_group(item, s.groups_by_id[gid], s.cols, layout)
        _snap(item, GROUP_FIELDS)

    # feature identity: first assay that names one, or the groups labelled feature_id
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
    d["literature_queries_ai"] = prop.get("literature_queries", [])
    d["literature"] = None
    d["clarifying_questions"] = prop.get("clarifying_questions", [])
    d["rejected"] = prop.get("rejected", [])
    refresh_samples(s, d)
    s.draft = d
    s.log("proposal", {"ai": d["ai"], "signature_hint": hint, "layout": d["layout"], "assays": d["assays"],
                       "groups": {gid: {k: it.get(k) for k in GROUP_FIELDS + ("confidence", "source", "validation")}
                                  for gid, it in d["groups"].items()},
                       "feature_identity": d["feature_identity"]})
    s.save()
    return d


def _attach_flags(s, gid, item):
    item["flag_values"] = flag_values(s, gid) if item.get("marks_rows_as_suspect") else None
    if item.get("marks_rows_as_suspect") and item.get("flagged_value") is None:
        item["flagged_value"] = _proposed_flag_value(item["flag_values"])


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
    """Apply a step decision atomically: on any error the draft is left untouched."""
    if step not in STEPS:
        raise StepError(f"Unknown step '{step}'.")
    original, groups, splits = s.draft, copy.deepcopy(s.groups), list(s.splits)
    s.draft = copy.deepcopy(original)
    try:
        return _confirm_step(s, step, decision, on_progress)
    except Exception:
        s.draft, s.groups, s.splits = original, groups, splits
        raise


def _clean(text, limit=300):
    return (str(text) if text is not None else "").strip()[:limit]


def _confirm_step(s, step, decision, on_progress=None):
    d = s.draft
    before = copy.deepcopy(d)
    reproposed = False
    roles_changed = False

    if step == "layout":
        lay = decision.get("layout")
        if lay not in VOCABULARY["layout"]:
            raise StepError("Choose one of the three layouts.")
        if lay != d["layout"]["value"]:
            build_draft(s, ai_on=d["ai"]["enabled"], fixed_layout=lay, on_progress=on_progress)
            d = s.draft
            reproposed = True
        d["layout"]["value"] = lay

    if "assays" in decision:
        new = decision["assays"]
        if not new:
            raise StepError("Describe at least one assay.")
        for k, a in enumerate(new):
            label = _clean(a.get("assay_label"), 120)
            if not label:
                raise StepError("Every assay needs a label.")
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
            old["assay_label"] = label
            old["omics_type"] = _clean(a.get("omics_type"), 80) or "unknown"
            old["source_software"] = _clean(a.get("source_software"), 120) or "unknown"
            if a.get("in_supported_scope") in VOCABULARY["in_supported_scope"]:
                old["in_supported_scope"] = a["in_supported_scope"]
            if "scope_reason" in a:
                old["scope_reason"] = _clean(a.get("scope_reason"))
        del d["assays"][len(new):]

    if "items" in decision:
        for ed in decision["items"]:
            gid = ed.get("group_id")
            if gid not in d["groups"]:
                raise StepError(f"Unknown group '{gid}'.")
            it = d["groups"][gid]
            for f in GROUP_FIELDS:
                if f in ed:
                    val = ed[f]
                    if f in ("label", "assay_label", "detail"):
                        val = _clean(val) or None if f != "label" else _clean(val)
                    if f in ("keep", "marks_rows_as_suspect"):
                        val = bool(val)
                    if f == "role" and val != it["role"]:
                        roles_changed = True
                    it[f] = val
            ai.normalize_item(it)
            if it["role"] == "value":
                if not it.get("assay_label"):
                    it["assay_label"] = d["assays"][0]["assay_label"]
                if it["assay_label"] not in [a["assay_label"] for a in d["assays"]]:
                    d["assays"].append(_snap({"assay_label": it["assay_label"], "omics_type": "unknown",
                                              "source_software": "unknown", "in_supported_scope": "unsure",
                                              "scope_reason": "", "source": "user", "confidence": 1.0,
                                              "evidence": "Added by you."}, ASSAY_FIELDS))
            if it.get("marks_rows_as_suspect") and it.get("flag_values") is None:
                it["flag_values"] = flag_values(s, gid)
            if it["role"] == "sample_metadata" and it.get("audit_kind") == "timepoint" and not it.get("detail") \
                    and s.groups_by_id[gid]["n_columns"] == 1:
                it["detail"] = timepoint_detail(s.cols.digests[s.groups_by_id[gid]["indices"][0]])
            it["validation"] = validate_group(it, s.groups_by_id[gid], s.cols, layout_of(d))
            if it["validation"]["status"] == "contradicted":
                raise StepError(f"{', '.join(s.groups_by_id[gid]['columns'][:2])}: "
                                + " ".join(it["validation"]["messages"]))

    if "feature_identity" in decision:
        gids = [g for g in decision["feature_identity"].get("group_ids", []) if g in d["groups"]]
        d["feature_identity"]["group_ids"] = gids
        d["feature_identity"]["composite"] = len(gids) > 1
        for gid, it in d["groups"].items():
            if gid in gids and it["role"] != "feature_id":
                it["role"] = "feature_id"
                ai.normalize_item(it)
                roles_changed = True
            elif gid not in gids and it["role"] == "feature_id":
                it["role"] = "feature_annotation"
                roles_changed = True
        d["feature_identity"]["validation"] = validate_feature_identity(d["feature_identity"], s.groups_by_id,
                                                                        s.cols, layout_of(d))

    if "sample_id_group" in decision:
        gid = decision["sample_id_group"]
        if gid is not None and gid not in d["groups"]:
            raise StepError(f"Unknown group '{gid}'.")
        for g2, it in d["groups"].items():
            if it["role"] == "sample_id" and g2 != gid:
                it["role"], it["audit_kind"] = "sample_metadata", "other"
        if gid:
            it = d["groups"][gid]
            it["role"] = "sample_id"
            ai.normalize_item(it)
            it["validation"] = validate_group(it, s.groups_by_id[gid], s.cols, layout_of(d))
        d["sample_id_group"]["value"] = gid

    if "sample_rules" in decision:
        for gid, rule in decision["sample_rules"].items():
            if gid in d["sample_rules"]:
                d["sample_rules"][gid]["strip_prefix"] = rule.get("strip_prefix", "")
                d["sample_rules"][gid]["strip_suffix"] = rule.get("strip_suffix", "")

    if "samples" in decision:
        for sid, ed in decision["samples"].items():
            if sid in d["samples"]:
                it = d["samples"][sid]
                if "label" in ed:
                    it["label"] = _clean(ed["label"], 120)
                if "is_study_sample" in ed:
                    it["is_study_sample"] = bool(ed["is_study_sample"])
                it["user_edited"] = True

    if "processing_history" in decision:
        ph = decision["processing_history"]
        for q, _ in HISTORY_QUESTIONS:
            a = ph.get(q) or {}
            if a.get("answer") not in VOCABULARY["yes_no_unsure"]:
                raise StepError("Answer every processing-history question (yes, no or not sure).")
            d["processing_history"][q] = {"answer": a["answer"], "note": _clean(a.get("note"))}
        d["software_and_version"] = _clean(ph.get("software_and_version"))
        d["history_notes"] = _clean(ph.get("notes"), 2000)

    if "metadata" in decision:
        apply_metadata_decision(s, d, decision["metadata"])

    _check_step(s, d, step)
    refresh_samples(s, d)
    d["steps"][step] = "confirmed"
    if roles_changed and not reproposed:
        idx = STEPS.index(step)
        for later in ("annotations", "values", "samples", "sample_info"):
            if STEPS.index(later) > idx and d["steps"][later] == "confirmed":
                d["steps"][later] = "pending"
    if reproposed:
        for st in STEPS[1:]:
            d["steps"][st] = "pending"
        d["processing_history"] = before["processing_history"]
        d["software_and_version"] = before.get("software_and_version", "")
        d["history_notes"] = before.get("history_notes", "")
        if before["steps"].get("history") == "confirmed":
            d["steps"]["history"] = "confirmed"
    for st in STEPS:
        if d["steps"][st] == "pending" and not step_applicable(s, d, st):
            d["steps"][st] = "not_applicable"
        elif d["steps"][st] == "not_applicable" and step_applicable(s, d, st):
            d["steps"][st] = "pending"
    s.log("step_confirmed", {"step": step, "decision": decision, "reproposed": reproposed,
                             "changes": _diff(before, d)})
    s.save()
    return {"draft": public_draft(s), "reproposed": reproposed}


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

def reconsider(s, gids, hint, on_progress=None):
    d = s.draft
    gids = [g for g in (gids if isinstance(gids, (list, tuple)) else [gids]) if g]
    unknown = [g for g in gids if g not in d["groups"]]
    if not gids or unknown:
        raise StepError(f"Unknown group(s): {', '.join(unknown) or 'none given'}.")
    ok, why = llm.available()
    if not (ok and d["ai"]["enabled"]):
        raise StepError("The AI is off or unavailable: " + (why or "turn it on to ask it to reconsider."))
    fixed = {"layout": d["layout"]["value"],
             "assays": [{f: a.get(f) for f in ASSAY_FIELDS} for a in d["assays"]],
             "confirmed_groups": {g: {k: it.get(k) for k in ("role", "label", "audit_kind", "keep")}
                                  for g, it in d["groups"].items() if g not in gids and it["role"] != UNRESOLVED},
             "user_feedback": _clean(hint, 1000),
             "instruction": ("The user disagrees with the previous proposal for these groups. Reconsider only "
                             "them, taking the user's feedback into account.")}
    prop, meta, digests = ai.propose(s.filename, s.sha, s.groups, s.hints, s.cols, fixed=fixed,
                                     group_ids=set(gids), log=s.log, on_progress=on_progress,
                                     signature_hint=d.get("signature_hint"))
    if meta.get("error"):
        raise StepError("The AI could not answer: " + meta["error"])
    labels = [a["assay_label"] for a in d["assays"]]
    split_ids = _apply_ai_splits(s, prop, d["layout"]["value"])
    changed = {}
    for gid in split_ids:
        new = _blank(prop["groups"][gid])
        new["hint"] = group_hint(s.groups_by_id[gid])
        _attach_flags(s, gid, new)
        d["groups"][gid] = _snap(new, GROUP_FIELDS)
        changed[gid] = {"old": None, "new": {k: new.get(k) for k in GROUP_FIELDS}}
    for gid in split_ids:  # the parent block's profile changed
        parent = s.groups_by_id[gid]["split_from"]
        if parent in d["groups"]:
            d["groups"][parent]["hint"] = group_hint(s.groups_by_id[parent])
    d["groups"] = {g["group_id"]: d["groups"][g["group_id"]] for g in s.groups}  # file order
    for gid in gids:
        new = prop["groups"].get(gid)
        if new is None:
            continue
        old = d["groups"][gid]
        new = _blank(new)
        new["hint"], new["keep"] = old.get("hint"), old.get("keep", True)
        if new["role"] == "value":
            if new.get("assay_label") not in labels:
                new["assay_label"] = old.get("assay_label") if old.get("assay_label") in labels else labels[0]
        _attach_flags(s, gid, new)
        _snap(new, GROUP_FIELDS)
        d["groups"][gid] = new
        changed[gid] = {"old": {k: old.get(k) for k in GROUP_FIELDS}, "new": {k: new.get(k) for k in GROUP_FIELDS}}
    s.log("reconsider", {"group_ids": gids, "user_feedback": hint, "changes": changed, "splits": split_ids})
    refresh_samples(s, d)
    s.save()
    return {"draft": public_draft(s), "digest": digests[0] if digests else None, "split_groups": split_ids,
            "questions": prop.get("clarifying_questions", [])}


# ---------------------------------------------------------------- literature (RAG)

def literature_description(s, d):
    """What the literature step works from: the (confirmed) assays and kept value
    blocks, with labels and computed statistics. No data values."""
    blocks = []
    for gid in value_blocks(s, d):
        it, g = d["groups"][gid], s.groups_by_id[gid]
        p = g.get("profile") or {}
        blocks.append({"group_id": gid, "assay_label": it.get("assay_label") or d["assays"][0]["assay_label"],
                       "label": it.get("label") or "", "n_columns": g["n_columns"],
                       "name": literature.block_name(it.get("label"), (g.get("pattern") or {}).get("text")),
                       "stats": {k: p.get(k) for k in ("median", "min", "max", "frac_zero", "frac_na",
                                                        "integer_valued", "log10_span")}})
    return {"layout": d["layout"]["value"],
            "assays": [{k: a.get(k) for k in ("assay_label", "omics_type", "source_software")} for a in d["assays"]],
            "value_blocks": blocks}


def suggested_queries(s, d):
    desc = literature_description(s, d)
    return list(dict.fromkeys((d.get("literature_queries_ai") or []) + literature.default_queries(desc)))[
        :literature.MAX_QUERIES]


def run_literature(s, queries=None, on_progress=None):
    d = s.draft
    if not literature.enabled():
        raise StepError("Literature search is turned off (PRISM_LITERATURE=off).")
    desc = literature_description(s, d)
    if not desc["value_blocks"]:
        raise StepError("Keep at least one value block first: the search is about the measurements.")
    queries = [literature._term(q) if q.count('"') % 2 else q.strip() for q in (queries or []) if q and q.strip()] \
        or suggested_queries(s, d)
    ok, _ = llm.available()
    try:
        rec = literature.run(desc, queries, ai_on=d["ai"]["enabled"] and ok, log=s.log, on_progress=on_progress)
    except literature.LiteratureError as e:
        raise StepError(str(e) + " Check your internet connection; you can continue without it.")
    d["literature"] = rec
    s.log("literature", {k: rec.get(k) for k in ("status", "queries", "summary", "blocks", "for_later_steps",
                                                   "rejected", "error")})
    s.save()
    return {"draft": public_draft(s)}


# ---------------------------------------------------------------- sample metadata file (samples in columns)

def _norm_id(x):
    x = re.sub(r"[\s_\-.]+", "", x.strip().lower())
    return re.sub(r"(?<![0-9])0+(?=[0-9])", "", x)


def upload_metadata(s, filename, raw):
    from .mock_llm import _NUM_RULES, _TEXT_RULES, _first
    t = parse_bytes(sanitize_filename(filename), raw)
    cols = Columns(t)
    data_ids = sample_ids(s, s.draft)
    best, best_hits = 0, -1
    for i in range(len(t["header"])):
        hits = len({cell(r, i).strip() for r in t["rows"]} & set(data_ids))
        if hits > best_hits:
            best, best_hits = i, hits
    s.metadata_table = {"table": t, "cols": cols}
    columns = []
    for i in range(len(t["header"])):
        dg = cols.digests[i]
        c = {"column": cols.labels[i], "index": i, "role": "sample_id" if i == best else "sample_metadata",
             "hint": group_hint({"kind": "single_column", "type": dg["type"], "profile": dg, "n_columns": 1}),
             "audit_kind": None, "label": "", "detail": None, "keep": True, "source": "computed"}
        if i != best:
            rule = _first(c["column"], _NUM_RULES if dg["type"] == "numeric" else _TEXT_RULES)
            if rule and rule[0] == "sample_metadata":
                c["audit_kind"], c["label"] = rule[2], rule[1]
            else:
                c["audit_kind"], c["label"] = "covariate", c["column"]
            if c["audit_kind"] == "timepoint":
                c["detail"] = timepoint_detail(dg)
        c["proposed"] = {"audit_kind": c["audit_kind"], "label": c["label"]}
        columns.append(c)
    meta = {"filename": sanitize_filename(filename), "id_column": t["header"][best], "columns": columns,
            "accepted_near_misses": [], "skipped": False,
            "report": match_report(data_ids, [cell(r, best).strip() for r in t["rows"]])}
    s.draft["metadata"] = meta
    s.log("metadata_upload", {"filename": meta["filename"], "id_column": meta["id_column"],
                              "report": {k: v for k, v in meta["report"].items() if k != "matched"}})
    s.save()
    return public_draft(s)


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


def apply_metadata_decision(s, d, md):
    if md.get("skip"):
        d["metadata"] = {"skipped": True}
        return
    meta = d.get("metadata")
    if not meta or meta.get("skipped"):
        raise StepError("Upload a metadata file or choose to skip.")
    valid = {(n["data_id"], n["metadata_id"]) for n in meta["report"]["near_misses"]}
    meta["accepted_near_misses"] = [p for p in md.get("accept_near_misses", []) if tuple(p) in valid]
    by_col = {c["column"]: c for c in meta["columns"]}
    for ed in md.get("columns", []):
        c = by_col.get(ed.get("column"))
        if not c or c["role"] == "sample_id":
            continue
        if ed.get("audit_kind") not in VOCABULARY["audit_kind"]:
            raise StepError(f"Choose an audit kind for metadata column '{c['column']}'.")
        c["audit_kind"] = ed["audit_kind"]
        c["label"] = _clean(ed.get("label", c["label"]))
        c["detail"] = _clean(ed.get("detail", c.get("detail"))) or None
        c["keep"] = bool(ed.get("keep", True))


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
        if it.get("marks_rows_as_suspect") and it.get("flag_values") and it.get("flagged_value") is not None:
            it["n_flagged"] = it["flag_values"].get(it["flagged_value"], 0)
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
    out["literature_queries_suggested"] = suggested_queries(s, d) if value_blocks(s, d) else []
    out["literature_enabled"] = literature.enabled()
    return out


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
        elif it.get("marks_rows_as_suspect") and it.get("flag_values") and it.get("flagged_value") is None:
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
