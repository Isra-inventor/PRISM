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

from . import ai, consistency, grouping, llm_providers as llm
from .format_detect import signature_hint, signature_prefill
from .mock_llm import expand_sample_rules
from .parsing import cell, is_missing, parse_bytes, sanitize_filename
from .profiling import Columns, apply_rule, layout_hints, make_group, shared_affixes
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
        self.affixes = shared_affixes(self.cols.labels)   # facts about the names, not groups
        self.hints = layout_hints(self.cols)
        self.groups = []   # proposed by the AI (or a known format / you) in build_draft
        self.draft = None
        self.digests = []
        self.metadata_table = None
        self.lock = threading.RLock()

    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "state.json").write_text(json.dumps({
            "sid": self.sid, "filename": self.filename, "groups": group_structure(self), "draft": self.draft,
            "digests": self.digests}, ensure_ascii=False), encoding="utf-8")

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

    d = {"schema_version": "0.2.1", "groups": {}, "sample_rules": {}, "samples": {}, "sample_rules_ai": [],
         "processing_history": {q: {"answer": None, "note": ""} for q, _ in HISTORY_QUESTIONS},
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
                     "chunk_size": ai.GROUPING_CHUNK_SIZE, "failed_chunks": prop.get("failed_chunks", []),
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
        assays = [{"assay_label": "assay 1", "omics_type": "unknown", "source_software": "unknown",
                   "in_supported_scope": "unsure", "scope_reason": "not described yet", "confidence": 0.0,
                   "evidence": "", "source": "none"}]
    d["assays"] = [_snap(a, ASSAY_FIELDS) for a in assays]

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
    for gid, item in d["groups"].items():
        if item["role"] != UNRESOLVED:
            item["validation"] = validate_group(item, s.groups_by_id[gid], s.cols, layout)
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
    d["clarifying_questions"] = [dict(q, group_id=to_gid(q.get("group_id"))) for q in prop.get("clarifying_questions", [])]
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
    """Apply a step decision atomically: on any error the draft is left untouched."""
    if step not in STEPS:
        raise StepError(f"Unknown step '{step}'.")
    original, groups = s.draft, copy.deepcopy(s.groups)
    s.draft = copy.deepcopy(original)
    try:
        return _confirm_step(s, step, decision, on_progress)
    except Exception:
        s.draft, s.groups = original, groups
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
                if "add_prefix" in rule:
                    d["sample_rules"][gid]["add_prefix"] = _clean(rule.get("add_prefix"), 60)

    if "derived" in decision:   # sample information parsed from column names (v2.3)
        der = d.get("derived_sample_metadata")
        if der:
            by = {c["name"]: c for c in der["columns"]}
            for ed in decision["derived"]:
                c = by.get(ed.get("name"))
                if not c:
                    continue
                if ed.get("audit_kind") not in VOCABULARY["audit_kind"]:
                    raise StepError(f"Choose an audit kind for '{c['name']}'.")
                c["audit_kind"] = ed["audit_kind"]
                c["label"] = _clean(ed.get("label", c.get("label")))
                c["keep"] = bool(ed.get("keep", True))

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

def _replace_groups(s, d, old_gids, pgroups, why):
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
    new_ids = _replace_groups(s, d, gids, prop["groups"], {"reconsider": hint})
    split_ids = [g for g in new_ids if g not in old_cols and s.groups_by_id[g].get("split_from")]
    s.log("reconsider", {"group_ids": gids, "user_feedback": hint, "result": new_ids,
                         "rejected": prop.get("rejected"), "merges": prop.get("merges_applied"),
                         "splits": prop.get("splits_applied")})
    s.save()
    return {"draft": public_draft(s), "digest": digests[0] if digests else None, "split_groups": split_ids,
            "new_groups": new_ids, "questions": prop.get("clarifying_questions", [])}


def retry_consolidation(s):
    """Run the cross-chunk consolidation again on the current groups; merges naming
    groups that exist are applied (as in the first proposal), others are rejected."""
    d = s.draft
    ok, why = llm.available()
    if not (ok and d["ai"]["enabled"]):
        raise StepError("The AI is off or unavailable: " + (why or ""))
    merges, err = ai.consolidate(s.cols, s.groups_by_id, d["groups"], s.sha, log=s.log)
    if err:
        raise StepError("The AI could not answer: " + err)
    applied, rejected = [], []
    for gids, reason in merges:
        gids = [g for g in dict.fromkeys(gids)]
        if len(gids) < 2 or any(g not in d["groups"] for g in gids):
            rejected.append({"group_ids": gids, "reason": "merge names groups that do not exist", "merge_reason": reason})
            continue
        keep = max(gids, key=lambda g: s.groups_by_id[g]["n_columns"])
        item = copy.deepcopy(d["groups"][keep])
        pg = {"m": {"indices": [i for g in gids for i in s.groups_by_id[g]["indices"]], "item": item,
                    "origin": "merged_consolidation", "merged_from": gids, "keep_id": keep}}
        _replace_groups(s, d, gids, pg, {"consolidation": gids, "reason": reason})
        applied.append({"group_ids": gids, "into": keep, "reason": reason})
    d["grouping"]["consolidation_error"] = None
    d["grouping"]["merges_applied"] = d["grouping"].get("merges_applied", []) + applied
    d["rejected"] = d.get("rejected", []) + rejected
    s.log("consolidation_retry", {"applied": applied, "rejected": rejected})
    s.save()
    return {"draft": public_draft(s), "merges": applied}


def merge_check(s, gids, hint=""):
    """'These groups are actually the same thing': ask the AI; nothing is applied."""
    d = s.draft
    gids = _check_gids(d, gids, 2)
    ok, why = llm.available()
    if not (ok and d["ai"]["enabled"]):
        return {"agrees": None, "reason": "The AI is off: you can merge them yourself.", "comment": ""}
    res = ai.merge_check(s.cols, s.groups_by_id, d["groups"], gids, _clean(hint, 1000), s.sha, log=s.log)
    s.log("merge_check", {"group_ids": gids, "user_hint": hint, "result": res})
    return res


def merge_groups(s, gids, why=None):
    """Your decision: merge these groups into one. The group with the most columns
    keeps its id and description; the result is re-validated (e.g. text + numbers
    cannot be a value block)."""
    d = s.draft
    gids = _check_gids(d, gids, 2)
    keep = max(gids, key=lambda g: s.groups_by_id[g]["n_columns"])
    item = copy.deepcopy(d["groups"][keep])
    item.update(source="user", evidence=f"Merged by you: {', '.join(gids)}." + (f" {why}" if why else ""))
    pg = {"m": {"indices": [i for g in gids for i in s.groups_by_id[g]["indices"]], "item": item,
                "origin": "merged_user", "merged_from": gids, "keep_id": keep}}
    new = _replace_groups(s, d, gids, pg, {"merge": gids, "reason": why})
    s.save()
    return {"draft": public_draft(s), "new_groups": new}


def split_columns(s, gid, columns):
    """Your decision: take these columns out of the group, one group each."""
    d = s.draft
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
                                    evidence=f"Taken out of {gid} by you: choose what it is.")}
    new = _replace_groups(s, d, [gid], pg, {"split": gid, "columns": columns})
    s.save()
    return {"draft": public_draft(s), "new_groups": new}


def group_columns(s, columns, why=None):
    """Your decision: these columns are one group (e.g. all columns sharing a name part).
    They leave their current groups; what is left of those groups stays as it was."""
    d = s.draft
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
    new = _replace_groups(s, d, touched, pg, {"group_columns": len(idx), "reason": why})
    s.save()
    return {"draft": public_draft(s), "new_groups": new}


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


# ---------------------------------------------------------------- "ask the AI to do it": plan, preview, apply

def _command_payload(s, d, instruction):
    def cols(g):
        c = g["columns"]
        return c if len(c) <= 4 else c[:3] + [f"... ({len(c) - 4} more)", c[-1]]
    groups = []
    for g in s.groups:
        it = d["groups"][g["group_id"]]
        x = {"group_id": g["group_id"], "n_columns": g["n_columns"], "columns": cols(g), "type": g["type"],
             "role": it["role"], "label": it.get("label"), "keep": it.get("keep", True)}
        for k in ("audit_kind", "assay_label"):
            if it.get(k):
                x[k] = it[k]
        if it.get("marks_rows_as_suspect"):
            x["marks_rows_as_suspect"] = True
        if g["type"] == "numeric":
            x["median"] = (g.get("profile") or {}).get("median")
        groups.append(x)
    ids = list(d["samples"])
    return {"instruction": instruction, "layout": d["layout"]["value"],
            "assays": [a["assay_label"] for a in d["assays"]], "groups": groups,
            "samples": {"n": len(ids), "first": [{"sample": x, "label": d["samples"][x].get("label"),
                                                   "is_study_sample": d["samples"][x].get("is_study_sample")}
                                                  for x in ids[:200]]}}


def _check_action(s, d, a):
    """-> (normalized action dict with a plain-words description, problems)."""
    from fnmatch import fnmatchcase
    x = {k: v for k, v in a.items() if v not in (None, [], "")}
    probs = []
    gids = [g for g in dict.fromkeys(a.get("group_ids") or [])]
    bad = [g for g in gids if g not in d["groups"]]
    if bad:
        probs.append(f"unknown group(s) {', '.join(bad[:5])} ignored")
    gids = [g for g in gids if g in d["groups"]]
    act = a["action"]
    names = lambda gs: ", ".join(s.groups_by_id[g]["columns"][0] for g in gs[:4]) + (f" … (+{len(gs) - 4})" if len(gs) > 4 else "")
    ncols = sum(s.groups_by_id[g]["n_columns"] for g in gids)
    if act == "set_samples":
        pats = a.get("samples") or []
        hit = [sid for sid in d["samples"] if any(sid == p or fnmatchcase(sid, p) for p in pats)]
        if not hit:
            probs.append("no sample matches")
        if a.get("label") is None and a.get("is_study_sample") is None:
            probs.append("nothing to change")
        what = ", ".join(filter(None, [f"label '{a['label']}'" if a.get("label") else None,
                                       None if a.get("is_study_sample") is None else
                                       ("study sample" if a["is_study_sample"] else "not a study sample")]))
        x.update(sample_ids=hit, description=f"{len(hit)} sample(s) ({', '.join(hit[:4])}{' …' if len(hit) > 4 else ''}): {what}")
        return x, probs
    if not gids:
        probs.append("no group to act on")
        return dict(x, group_ids=[], description=act), probs
    x["group_ids"] = gids
    if act == "set_keep":
        if a.get("keep") is None:
            probs.append("keep is missing")
        ids_ = [g for g in gids if d["groups"][g]["role"] in ("feature_id", "sample_id")]
        if ids_ and a.get("keep") is False:
            probs.append(f"identifier columns cannot be excluded: {names(ids_)}")
            x["group_ids"] = gids = [g for g in gids if g not in ids_]
            ncols = sum(s.groups_by_id[g]["n_columns"] for g in gids)
        verb = "leave out of the outputs" if a.get("keep") is False else "keep in the outputs"
        x["description"] = f"{verb}: {len(gids)} group(s), {ncols} column(s) — {names(gids)}"
    elif act == "set_role":
        if a.get("role") not in VOCABULARY["column_role"] or a.get("role") == UNRESOLVED:
            probs.append(f"'{a.get('role')}' is not an allowed role")
        else:
            ok = []
            for g in gids:
                v = validate_group(dict(d["groups"][g], role=a["role"]), s.groups_by_id[g], s.cols, layout_of(d))
                if v["status"] == "contradicted":
                    probs.append(f"{names([g])}: " + " ".join(v["messages"]))
                else:
                    ok.append(g)
            x["group_ids"] = gids = ok
        x["description"] = f"role '{pretty(a.get('role'))}' for {len(gids)} group(s): {names(gids)}"
    elif act == "set_label":
        if not (a.get("label") or "").strip():
            probs.append("label is empty")
        x["description"] = f"label '{a.get('label')}' for {len(gids)} group(s): {names(gids)}"
    elif act == "set_audit_kind":
        if a.get("audit_kind") not in VOCABULARY["audit_kind"]:
            probs.append(f"'{a.get('audit_kind')}' is not an allowed audit kind")
        not_meta = [g for g in gids if d["groups"][g]["role"] != "sample_metadata"]
        if not_meta:
            probs.append(f"not sample information (role unchanged): {names(not_meta)}")
            x["group_ids"] = gids = [g for g in gids if g not in not_meta]
        x["description"] = f"audit kind '{pretty(a.get('audit_kind'))}' for {len(gids)} group(s): {names(gids)}"
    elif act == "set_assay":
        if not (a.get("assay_label") or "").strip():
            probs.append("assay label is empty")
        x["description"] = f"assay '{a.get('assay_label')}' for {len(gids)} group(s): {names(gids)}"
    elif act == "set_suspect":
        x["description"] = (f"{'mark' if a.get('marks_rows_as_suspect') else 'unmark'} as marking rows suspect: "
                            f"{names(gids)}")
    elif act == "merge":
        if len(gids) < 2:
            probs.append("a merge needs at least two existing groups")
        x["description"] = f"merge {len(gids)} groups ({ncols} columns) into one: {names(gids)}"
    elif act == "split":
        g = gids[0]
        cols = [c for c in a.get("columns") or [] if c in s.groups_by_id[g]["columns"]]
        if not cols:
            probs.append("no column of that group named")
        elif len(cols) == s.groups_by_id[g]["n_columns"]:
            probs.append("that is every column of the group")
        x.update(group_ids=[g], columns=cols, description=f"take {len(cols)} column(s) out of {names([g])}: {', '.join(cols[:4])}")
    else:
        probs.append(f"unknown action '{act}'")
        x["description"] = act
    return x, probs


def pretty(x):
    return str(x or "").replace("_", " ")


def plan_command(s, instruction):
    """Ask the AI to turn an instruction into actions. Nothing is applied: you get a preview."""
    d = s.draft
    instruction = _clean(instruction, 1000)
    if not instruction:
        raise StepError("Write what you want the AI to do.")
    ok, why = llm.available()
    if not (ok and d["ai"]["enabled"]):
        raise StepError("The AI is off or unavailable: " + (why or "turn it on to give it instructions."))
    resp, err = ai.command(_command_payload(s, d, instruction), s.sha, log=s.log)
    if resp is None:
        raise StepError("The AI could not answer: " + (err or "no answer"))
    actions = []
    for k, a in enumerate(resp.actions):
        x, probs = _check_action(s, d, a.model_dump())
        usable = not [p for p in probs if p.startswith(("unknown action", "no group", "no sample", "nothing to",
                                                         "keep is", "label is", "assay label is", "a merge needs",
                                                         "no column", "that is every", "'"))] and \
            (x.get("group_ids") or x.get("sample_ids"))
        actions.append(dict(x, index=k, problems=probs, usable=bool(usable)))
    cmd = {"id": uuid.uuid4().hex[:8], "instruction": instruction, "reply": resp.reply,
           "not_possible": resp.not_possible, "actions": actions}
    d["pending_command"] = cmd
    s.log("ai_command_plan", cmd)
    s.save()
    return {"draft": public_draft(s), "command": cmd}


def apply_command(s, cmd_id, accept=None):
    """Apply the accepted actions of the pending instruction (your confirmation)."""
    d = s.draft
    cmd = d.get("pending_command")
    if not cmd or cmd["id"] != cmd_id:
        raise StepError("That instruction is no longer pending; ask again.")
    chosen = [a for a in cmd["actions"] if a["usable"] and (accept is None or a["index"] in accept)]
    done, skipped = [], []
    for a in chosen:
        x, probs = _check_action(s, d, a)          # ids may have changed by an earlier merge / split
        gids = x.get("group_ids") or []
        if not (gids or x.get("sample_ids")):
            skipped.append({"action": a["description"], "why": "; ".join(probs) or "nothing left to change"})
            continue
        steps = {step_for_group(s, d, g) for g in gids}
        act = a["action"]
        if act == "merge":
            merge_groups(s, gids, f"Instruction: {cmd['instruction']}")
        elif act == "split":
            split_columns(s, gids[0], x["columns"])
        elif act == "set_samples":
            for sid in x["sample_ids"]:
                it = d["samples"][sid]
                if a.get("label"):
                    it["label"] = _clean(a["label"], 120)
                if a.get("is_study_sample") is not None:
                    it["is_study_sample"] = bool(a["is_study_sample"])
                it["user_edited"] = True
            steps = {"samples"}
        else:
            for g in gids:
                it = d["groups"][g]
                before = copy.deepcopy(it)
                if act == "set_keep":
                    it["keep"] = bool(a["keep"])
                    if not a["keep"] and it["role"] == UNRESOLVED:
                        it["role"] = "ignore"
                elif act == "set_role":
                    it["role"] = a["role"]
                elif act == "set_label":
                    it["label"] = _clean(a["label"])
                elif act == "set_audit_kind":
                    it["audit_kind"] = a["audit_kind"]
                elif act == "set_assay":
                    it["assay_label"] = _clean(a["assay_label"], 120)
                elif act == "set_suspect":
                    it["marks_rows_as_suspect"] = bool(a.get("marks_rows_as_suspect"))
                ai.normalize_item(it)
                if it["role"] == "value" and not it.get("assay_label"):
                    it["assay_label"] = d["assays"][0]["assay_label"]
                if it["role"] == "value" and it["assay_label"] not in [z["assay_label"] for z in d["assays"]]:
                    d["assays"].append(_snap({"assay_label": it["assay_label"], "omics_type": "unknown",
                                              "source_software": "unknown", "in_supported_scope": "unsure",
                                              "scope_reason": "", "source": "user", "confidence": 1.0,
                                              "evidence": "Added by an instruction."}, ASSAY_FIELDS))
                _attach_flags(s, g, it)
                it["validation"] = validate_group(it, s.groups_by_id[g], s.cols, layout_of(d))
                if it["validation"]["status"] == "contradicted":
                    d["groups"][g] = before
                    skipped.append({"action": a["description"], "why": f"{g}: " + " ".join(it["validation"]["messages"])})
                    continue
                steps.add(step_for_group(s, d, g))
        for st in steps:
            if d["steps"].get(st) == "confirmed":
                d["steps"][st] = "pending"
        done.append(a["description"])
    d["pending_command"] = None
    refresh_samples(s, d)
    s.log("ai_command_applied", {"instruction": cmd["instruction"], "applied": done, "skipped": skipped})
    s.save()
    return {"draft": public_draft(s), "applied": done, "skipped": skipped}


def discard_command(s):
    s.draft["pending_command"] = None
    s.save()
    return {"draft": public_draft(s)}


# ---------------------------------------------------------------- consistency (v2.3): flags you confirm

def consistency_report(s, d):
    """Deterministic checks: near-identical identifier-named blocks, and sample-ID
    collisions between blocks of one assay (samples in columns)."""
    dismissed = set(d.get("consistency_dismissed", []))
    flags = consistency.block_flags(s.groups_by_id, d["groups"], dismissed)
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
    return {"block_flags": flags, "sample_collisions": collisions, "sample_structure": structure}


def resolve_consistency(s, action, key=None, group_ids=None, labels=None):
    d = s.draft
    if action == "dismiss":
        if not any(f["key"] == key for f in consistency_report(s, d)["block_flags"]):
            raise StepError("This flag no longer applies.")
        d.setdefault("consistency_dismissed", []).append(key)
        s.log("consistency", {"action": "dismiss", "key": key})
    elif action == "one_block":
        flag = next((f for f in consistency_report(s, d)["block_flags"] if f["key"] == key), None)
        if flag is None:
            raise StepError("This flag no longer applies.")
        _treat_as_one_block(s, d, flag)
    elif action in ("full_names", "labels"):
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
        s.log("consistency", {"action": action, "group_ids": gids, "labels": labels})
    else:
        raise StepError(f"Unknown action '{action}'.")
    if d["steps"].get("samples") == "confirmed":
        d["steps"]["samples"] = "pending"
    refresh_samples(s, d)
    s.save()
    return {"draft": public_draft(s)}


DERIVED_COLUMNS = [
    {"name": "subject_code", "audit_kind": "subject_id", "label": "code at the start of the column name", "keep": True},
    {"name": "name_suffix", "audit_kind": "timepoint", "label": "rest of the column name (e.g. a time point)", "keep": True},
]


def _treat_as_one_block(s, d, flag):
    """Your confirmation: one block; the former block boundaries (subject codes) and the
    rest of each column name become sample information; sample IDs are the full names."""
    gids = flag["group_ids"]
    codes = {g: consistency.name_code(s.groups_by_id[g]) for g in gids}
    der = d.get("derived_sample_metadata") or {"source": "column names of merged blocks",
                                              "columns": [dict(c) for c in DERIVED_COLUMNS], "values": {}}
    for g in gids:
        code = codes[g]
        for i in s.groups_by_id[g]["indices"]:
            name = s.table["header"][i]
            rest = name[len(code):].strip(consistency._SEP) if code and name.startswith(code) else ""
            der["values"][name] = {"subject_code": code or "", "name_suffix": rest}
    d["derived_sample_metadata"] = der
    keep = max(gids, key=lambda g: s.groups_by_id[g]["n_columns"])
    item = copy.deepcopy(d["groups"][keep])
    item.update(source="user", evidence=f"You confirmed these {len(gids)} blocks are one measurement "
                                        f"({', '.join(c or '?' for c in codes.values())} are subjects / samples).")
    pg = {"m": {"indices": [i for g in gids for i in s.groups_by_id[g]["indices"]], "item": item,
                "origin": "merged_consistency", "merged_from": gids, "keep_id": keep}}
    new = _replace_groups(s, d, gids, pg, {"consistency_one_block": gids})
    for g in new:
        d["sample_rules"][g] = _snap({"strip_prefix": "", "strip_suffix": "", "source": "user"},
                                     ("strip_prefix", "strip_suffix"))
    s.log("consistency", {"action": "one_block", "group_ids": gids, "codes": codes, "into": new})


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
    out["name_parts"] = name_parts(s)
    out["ai_ungrouped"] = [g["group_id"] for g in s.groups if g.get("origin") == "ai_unavailable"]
    out["consistency"] = consistency_report(s, d)
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
    rep = consistency_report(s, d)
    for f in rep["block_flags"]:
        out.append({"step": "values", "flag": f["key"],
                    "what": f"{len(f['group_ids'])} blocks look statistically identical ({', '.join(c or '?' for c in f['codes'])}): "
                            "one measurement across subjects, or really different measurements?"})
    for c in rep["sample_collisions"]:
        out.append({"step": "samples", "what": c["message"]})
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
