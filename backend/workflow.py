"""Session workflow: build the draft schema, apply wizard steps, track provenance.

The draft is the single source of truth for a session. The frontend renders
it; every wizard step sends a decision that is applied here, validated and
logged. Provenance is derived by comparing the current value with the value
first proposed and who proposed it.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import threading
import uuid
from collections import Counter, OrderedDict
from pathlib import Path

from . import ai, llm_providers as llm
from .format_detect import signature_prefill
from .mock_llm import expand_sample_types
from .parsing import cell, is_missing, parse_bytes, sanitize_filename
from .profiling import Columns, apply_rule, build_groups, layout_hints
from .schema import HISTORY_QUESTIONS, UNRESOLVED, VOCABULARY, kinds_for_role
from .session_log import log_event
from .validation import timepoint_detail, uniqueness, validate_feature_identity, validate_group

import os

SESSIONS_DIR = Path(os.environ.get("PRISM_SESSIONS_DIR", Path(__file__).parent / "sessions"))
STEPS = ["layout", "feature_id", "annotations", "values", "samples", "sample_info", "history", "review"]
GROUP_FIELDS = ("role", "kind", "detail", "measurement_type", "scale", "omics_type", "label", "block_role", "keep")
_SAMPLE_TYPE_WORDS = [("qc", "qc"), ("pool", "pool"), ("blank", "blank"), ("buffer", "blank"),
                      ("calib", "calibrator"), ("std", "calibrator")]
_MAX_SESSIONS = 20
_sessions = OrderedDict()
_lock = threading.Lock()


class StepError(Exception):
    pass


# ---------------------------------------------------------------- provenance

def provenance(item, fields):
    """signature | computed | ai_proposed_confirmed | ai_proposed_corrected | user_set"""
    src = item.get("source") or "none"
    proposed = item.get("proposed") or {}
    changed = any(item.get(f) != proposed.get(f) for f in fields if f in proposed or f in item)
    if src == "signature":
        return "user_set" if changed else "signature"
    if src == "computed":
        return "user_set" if changed else "computed"
    if src == "ai":
        return "ai_proposed_corrected" if changed else "ai_proposed_confirmed"
    return "user_set"


def _snap(item, fields):
    item["proposed"] = {f: copy.deepcopy(item.get(f)) for f in fields}
    return item


FACT_FIELDS = ("value",)
FI_FIELDS = ("group_ids", "composite")


# ---------------------------------------------------------------- session

class Session:
    def __init__(self, sid, filename, raw):
        self.sid = sid
        self.filename = filename
        self.dir = SESSIONS_DIR / sid
        self.table = parse_bytes(filename, raw)
        self.sha = self.table["sha256"]
        self.cols = Columns(self.table)
        self.base_groups = build_groups(self.cols)
        self.splits = []  # accepted split decisions, re-applied on reload
        self.groups = copy.deepcopy(self.base_groups)
        self.hints = layout_hints(self.cols, self.groups)
        self.draft = None
        self.digests = []
        self.metadata_table = None
        self.lock = threading.RLock()

    # -- persistence
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

    def group_of_column(self, label):
        for g in self.groups:
            if label in g["columns"]:
                return g
        return None


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
        s.metadata_table = {"table": t, "cols": Columns(t), "groups": []}
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


# ---------------------------------------------------------------- hints for manual mode

def group_hint(g, layout=None):
    p = g.get("profile") or {}
    if g["kind"] == "numeric_block":
        pat = (g.get("pattern") or {}).get("text")
        base = f"{g['n_columns']} numeric columns" + (f" sharing '{pat.strip()}'" if pat else " with similar values")
        return base + f"; median {p.get('median')}, {int((p.get('frac_zero') or 0) * 100)}% zeros."
    if g["type"] == "numeric":
        return (f"Numeric column: median {p.get('median')}, range {p.get('min')}–{p.get('max')}"
                + (", whole numbers" if p.get("integer_valued") else "") + ".")
    vals = p.get("values")
    if vals:
        return f"Text with {p.get('n_unique')} distinct values: " + ", ".join(v["value"] for v in vals[:6]) + "."
    return f"Text with {p.get('n_unique')} distinct values ({int((p.get('unique_ratio') or 0) * 100)}% unique)."


def layout_hint_text(h):
    if h["long_format_pattern"]:
        return "Repeated identifier columns and a single numeric column: this may be a long table."
    if h["largest_numeric_block_columns"]:
        r = h["rows_to_block_columns_ratio"]
        if r is not None and r >= 1:
            return (f"{h['n_rows']} rows and a numeric block of {h['largest_numeric_block_columns']} columns: "
                    "features usually outnumber samples, so rows are probably features (samples in columns).")
        return (f"Only {h['n_rows']} rows but a numeric block of {h['largest_numeric_block_columns']} columns: "
                "rows are probably samples (samples in rows).")
    return "No large block of numeric columns was found."


# ---------------------------------------------------------------- building the draft

def build_draft(s, ai_on=True, fixed_layout=None, on_progress=None):
    """Signature pre-fill + AI proposal (or manual hints) -> a fresh draft."""
    sig = signature_prefill(s.table["header"], s.groups)
    prop, meta, digests = None, {"error": None}, []
    ok, why = llm.available()
    use_ai = ai_on and ok
    if use_ai:
        fixed = {}
        if fixed_layout:
            fixed["layout"] = fixed_layout
        if sig:
            fixed.update({"layout": fixed_layout or sig["layout"]["value"], "omics_type": sig["omics_type"]["value"],
                          "source_software": sig["source_software"]["value"]})
        prop, meta, digests = ai.propose(s.filename, s.sha, s.groups, s.hints, s.cols, fixed=fixed,
                                         log=s.log, on_progress=on_progress)
        if prop.get("splits"):
            for sp in prop["splits"]:
                new_ids = _apply_split(s, sp)
                if new_ids:
                    s.log("split_applied", {"split": sp, "new_groups": new_ids})
                    for gid in new_ids:
                        g = s.groups_by_id[gid]
                        item = {"role": sp.get("role") or UNRESOLVED, "kind": sp.get("kind"), "detail": None,
                                "measurement_type": None, "scale": None, "omics_type": None, "label": None,
                                "confidence": 0.7, "source": "ai",
                                "evidence": f"Split out of the block on the AI's suggestion: {sp.get('evidence', '')}"}
                        prop["groups"][gid] = ai.finalize_item(item, g, s.cols, fixed_layout)
    s.digests = digests
    d = _empty_draft(s)
    d["ai"] = {"enabled": bool(ai_on), "available": ok, "unavailable_reason": None if ok else why,
               "provider": meta.get("provider") or llm.provider_name(), "model": meta.get("model"),
               "models_used": meta.get("models_used", []), "prompt_version": ai.PROMPT_VERSION,
               "temperature": llm.TEMPERATURE, "error": meta.get("error"), "cached": meta.get("cached", False)}
    d["signature"] = {"name": sig["signature"], "platform": sig["platform"], "also_matched": sig["also_matched"]} \
        if sig else None

    # file-level facts: signature > fixed layout > AI > unresolved (with a deterministic hint)
    for key in ("layout", "omics_type", "source_software"):
        item = None
        if sig and key in sig and not (key == "layout" and fixed_layout):
            item = dict(sig[key])
        elif key == "layout" and fixed_layout:
            item = {"value": fixed_layout, "confidence": 1.0, "evidence": "Confirmed by you.", "source": "user"}
        elif prop and key in prop:
            item = dict(prop[key])
        if item is None:
            item = {"value": UNRESOLVED if key == "layout" else "unknown", "confidence": 0.0, "evidence": "",
                    "source": "none"}
        item.setdefault("validation", {"status": "ok", "messages": []})
        d[key] = _snap(item, FACT_FIELDS)
    d["layout"]["hint"] = layout_hint_text(s.hints)

    # groups
    for g in s.groups:
        gid = g["group_id"]
        item = None
        if sig and gid in sig["groups"]:
            item = dict(sig["groups"][gid])
            ai_item = (prop or {}).get("groups", {}).get(gid) or {}
            for f in ("scale", "omics_type", "measurement_type", "label"):
                if item.get("role") == "value" and not item.get(f) and ai_item.get(f):
                    item[f] = ai_item[f]
            item = ai.finalize_item(_blank(item), g, s.cols, sig["layout"]["value"])
        elif prop and gid in prop["groups"]:
            item = _blank(prop["groups"][gid])
        else:
            item = ai.unresolved_item("AI is off: choose this role yourself." if not use_ai else
                                      "No proposal for this group.")
        item["hint"] = group_hint(g)
        item.setdefault("keep", True)
        if item["role"] == "value" and not item.get("omics_type"):
            item["omics_type"] = d["omics_type"]["value"] if d["omics_type"]["value"] in VOCABULARY["omics_type"] \
                else "unknown"
        d["groups"][gid] = item
    _assign_block_roles(d, s)
    lay = d["layout"]["value"]
    for gid, item in d["groups"].items():
        if item["role"] != UNRESOLVED:  # re-check with the final layout (keeps earlier contradictions)
            item["validation"] = validate_group(item, s.groups_by_id[gid], s.cols, lay)
        _snap(item, GROUP_FIELDS)

    # feature identity
    fi = None
    if sig and sig.get("feature_identity"):
        fi = dict(sig["feature_identity"])
    elif prop and prop.get("feature_identity") and prop["feature_identity"]["group_ids"]:
        fi = dict(prop["feature_identity"])
    if fi is None:
        fid_groups = [gid for gid, it in d["groups"].items() if it["role"] == "feature_id"]
        fi = {"group_ids": fid_groups, "composite": len(fid_groups) > 1, "confidence": 0.5 if fid_groups else 0.0,
              "evidence": "Groups labelled as feature ID." if fid_groups else "", "source": "computed" if fid_groups else "none"}
    fi["validation"] = validate_feature_identity(fi, s.groups_by_id, s.cols, d["layout"]["value"])
    for gid in fi["group_ids"]:
        if gid in d["groups"] and d["groups"][gid]["role"] in (UNRESOLVED, "feature_annotation"):
            d["groups"][gid]["role"] = "feature_id"
            d["groups"][gid]["proposed"]["role"] = "feature_id"
    d["feature_identity"] = _snap(fi, FI_FIELDS)

    # sample id column (samples in rows / long)
    sid_groups = [gid for gid, it in d["groups"].items() if it["role"] == "sample_id"]
    d["sample_id_group"] = _snap({"value": sid_groups[0] if sid_groups else None,
                                  "source": d["groups"][sid_groups[0]]["source"] if sid_groups else "none"},
                                 FACT_FIELDS)
    # sample name rules for value blocks
    for g in s.groups:
        if g.get("sample_id_rule") is not None:
            d["sample_rules"][g["group_id"]] = _snap(dict(g["sample_id_rule"], source="computed"),
                                                     ("strip_prefix", "strip_suffix"))
    d["sample_type_rules"] = (prop or {}).get("sample_types", [])
    d["clarifying_questions"] = (prop or {}).get("clarifying_questions", [])
    d["rejected"] = (prop or {}).get("rejected", [])
    refresh_sample_types(s, d)
    s.draft = d
    s.log("proposal", {"ai": d["ai"], "signature": d["signature"],
                       "groups": {gid: {k: it.get(k) for k in GROUP_FIELDS + ("confidence", "source", "validation")}
                                  for gid, it in d["groups"].items()},
                       "layout": d["layout"], "feature_identity": d["feature_identity"]})
    s.save()
    return d


def _blank(item):
    base = {"role": UNRESOLVED, "kind": None, "detail": None, "measurement_type": None, "scale": None,
            "omics_type": None, "label": None, "confidence": 0.0, "evidence": "", "source": "none",
            "validation": {"status": "ok", "messages": []}}
    base.update({k: v for k, v in item.items() if v is not None or k not in base})
    return base


def _empty_draft(s):
    return {
        "schema_version": "0.2",
        "groups": {},
        "sample_rules": {},
        "sample_types": {},
        "sample_type_rules": [],
        "processing_history": {q: {"answer": None, "note": ""} for q, _ in HISTORY_QUESTIONS},
        "software_and_version": "",
        "history_notes": "",
        "metadata": None,
        "steps": {st: "pending" for st in STEPS},
    }


def _assign_block_roles(d, s):
    """At most one primary value block per omics type: keep a signature's choice,
    otherwise the widest block becomes primary (computed); others auxiliary."""
    by_omics = {}
    for gid, it in d["groups"].items():
        if it["role"] == "value":
            by_omics.setdefault(it.get("omics_type") or "unknown", []).append(gid)
    for omics, gids in by_omics.items():
        primaries = [g for g in gids if d["groups"][g].get("block_role") == "primary"]
        if not primaries:
            best = max(gids, key=lambda g: (s.groups_by_id[g]["n_columns"], -min(s.groups_by_id[g]["indices"])))
            primaries = [best]
            d["groups"][best]["block_role"] = "primary"
            if d["groups"][best].get("source") != "signature":
                d["groups"][best]["block_role_source"] = "computed"
        for g in gids:
            if g not in primaries[:1]:
                if d["groups"][g].get("block_role") in (None, "primary"):
                    d["groups"][g]["block_role"] = "auxiliary"


# ---------------------------------------------------------------- samples

def layout_of(d):
    return d["layout"]["value"]


def primary_blocks(s, d):
    return [gid for gid, it in d["groups"].items() if it["role"] == "value" and it.get("block_role") == "primary"]


def sample_ids(s, d, gid=None):
    """Sample IDs in file order, as the canonical outputs will name them."""
    lay = layout_of(d)
    if lay == "samples_in_columns":
        out = []
        blocks = [gid] if gid else primary_blocks(s, d)
        for b in blocks:
            g = s.groups_by_id.get(b)
            if not g:
                continue
            rule = d["sample_rules"].get(b) or {"strip_prefix": "", "strip_suffix": ""}
            names = [apply_rule(s.table["header"][i], rule) for i in g["indices"]]
            out.extend(n for n in names if n not in out)
        return out
    sg = (d.get("sample_id_group") or {}).get("value")
    if not sg or sg not in s.groups_by_id:
        return [f"row_{k + 1}" for k in range(len(s.table["rows"]))] if lay == "samples_in_rows" else []
    i = s.groups_by_id[sg]["indices"][0]
    vals = [cell(r, i).strip() for r in s.table["rows"]]
    if lay == "long":
        seen = []
        for v in vals:
            if v not in seen:
                seen.append(v)
        return seen
    return vals


def refresh_sample_types(s, d):
    """(Re)compute proposed sample types for the current sample IDs, keeping
    anything the user already set."""
    ids = sample_ids(s, d)
    rules = expand_sample_types(d.get("sample_type_rules", []), ids)
    type_col = None
    if layout_of(d) in ("samples_in_rows", "long"):
        for gid, it in d["groups"].items():
            if it["role"] == "sample_metadata" and it.get("kind") == "sample_type":
                type_col = s.groups_by_id[gid]["indices"][0]
    col_vals = {}
    if type_col is not None and layout_of(d) == "samples_in_rows":
        for r, sid in zip(s.table["rows"], ids):
            col_vals[sid] = cell(r, type_col).strip()
    old = d.get("sample_types", {})
    new = {}
    for sid in ids:
        if sid in old and old[sid].get("user_edited"):
            new[sid] = old[sid]
            continue
        t, src, ev, conf = None, "computed", "", 0.5
        if sid in col_vals:
            v = col_vals[sid].lower()
            t = {"sample": "study", "study": "study", "qc": "qc", "pool": "pool", "blank": "blank",
                 "buffer": "blank", "calibrator": "calibrator", "calib": "calibrator"}.get(v)
            if t:
                ev = f"Sample type column says '{col_vals[sid]}'."
        if t is None and sid in rules:
            t, src, ev, conf = rules[sid]["type"], "ai", rules[sid]["evidence"], rules[sid]["confidence"]
        if t is None:
            low = sid.lower()
            for w, typ in _SAMPLE_TYPE_WORDS:
                if re.search(rf"(^|[^a-z]){w}", low):
                    t, ev = typ, f"Sample name contains '{w}'."
                    break
        if t is None:
            t, ev = "study", "No QC / blank / pool / calibrator hint in the name."
        new[sid] = _snap({"type": t, "source": src, "evidence": ev, "confidence": conf}, ("type",))
    d["sample_types"] = new


# ---------------------------------------------------------------- confirming steps

def step_applicable(s, d, step):
    lay = layout_of(d)
    if step == "annotations":
        return lay != "samples_in_rows" or any(it["role"] == "feature_annotation" for it in d["groups"].values())
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


def _confirm_step(s, step, decision, on_progress=None):
    d = s.draft
    before = copy.deepcopy(d)
    reproposed = False
    roles_changed = False

    if step == "layout":
        lay = decision.get("layout")
        if lay not in VOCABULARY["layout"]:
            raise StepError("Choose one of the three layouts.")
        if decision.get("omics_type") not in VOCABULARY["omics_type"]:
            raise StepError("Choose an omics type.")
        if lay != d["layout"]["value"]:
            # re-propose everything with the layout fixed as confirmed
            build_draft(s, ai_on=d["ai"]["enabled"], fixed_layout=lay, on_progress=on_progress)
            d = s.draft
            reproposed = True
        d["layout"]["value"] = lay
        d["omics_type"]["value"] = decision["omics_type"]
        d["source_software"]["value"] = (decision.get("source_software") or "unknown").strip() or "unknown"
        for gid, it in d["groups"].items():
            if it["role"] == "value" and (it.get("omics_type") in (None, "unknown")):
                it["omics_type"] = decision["omics_type"]

    if "items" in decision:
        for ed in decision["items"]:
            gid = ed.get("group_id")
            if gid not in d["groups"]:
                raise StepError(f"Unknown group '{gid}'.")
            it = d["groups"][gid]
            for f in GROUP_FIELDS:
                if f in ed:
                    if f == "role" and ed[f] != it["role"]:
                        roles_changed = True
                    it[f] = ed[f]
            if it["role"] not in ("feature_annotation", "sample_metadata"):
                it["kind"] = None
            elif it.get("kind") not in kinds_for_role(it["role"]):
                it["kind"] = None
            if it["role"] != "value":
                for f in ("measurement_type", "scale", "block_role"):
                    it[f] = None
            elif not it.get("block_role"):
                it["block_role"] = "auxiliary"
            if it["role"] == "sample_metadata" and it.get("kind") == "timepoint" and not it.get("detail"):
                it["detail"] = timepoint_detail(s.cols.digests[s.groups_by_id[gid]["indices"][0]]) \
                    if s.groups_by_id[gid]["n_columns"] == 1 else None
            it["validation"] = validate_group(it, s.groups_by_id[gid], s.cols, layout_of(d))

    if "feature_identity" in decision:
        fi = decision["feature_identity"]
        gids = [g for g in fi.get("group_ids", []) if g in d["groups"]]
        d["feature_identity"]["group_ids"] = gids
        d["feature_identity"]["composite"] = len(gids) > 1
        for gid, it in d["groups"].items():
            if gid in gids and it["role"] != "feature_id":
                it["role"], it["kind"] = "feature_id", None
                roles_changed = True
            elif gid not in gids and it["role"] == "feature_id":
                it["role"], it["kind"] = "feature_annotation", "other_annotation"
                roles_changed = True
        d["feature_identity"]["validation"] = validate_feature_identity(d["feature_identity"], s.groups_by_id, s.cols,
                                                                        layout_of(d))

    if "sample_id_group" in decision:
        gid = decision["sample_id_group"]
        if gid is not None and gid not in d["groups"]:
            raise StepError(f"Unknown group '{gid}'.")
        for g2, it in d["groups"].items():
            if it["role"] == "sample_id" and g2 != gid:
                it["role"], it["kind"] = "sample_metadata", "other_sample_metadata"
        if gid:
            d["groups"][gid]["role"], d["groups"][gid]["kind"] = "sample_id", None
            d["groups"][gid]["validation"] = validate_group(d["groups"][gid], s.groups_by_id[gid], s.cols,
                                                            layout_of(d))
        d["sample_id_group"]["value"] = gid

    if "sample_rules" in decision:
        for gid, rule in decision["sample_rules"].items():
            if gid in d["sample_rules"]:
                d["sample_rules"][gid]["strip_prefix"] = rule.get("strip_prefix", "")
                d["sample_rules"][gid]["strip_suffix"] = rule.get("strip_suffix", "")

    if "sample_types" in decision:
        for sid, t in decision["sample_types"].items():
            if t not in VOCABULARY["sample_type"]:
                raise StepError(f"'{t}' is not a sample type.")
            if sid in d["sample_types"] and d["sample_types"][sid]["type"] != t:
                d["sample_types"][sid]["type"] = t
                d["sample_types"][sid]["user_edited"] = True

    if "processing_history" in decision:
        ph = decision["processing_history"]
        for q, _ in HISTORY_QUESTIONS:
            a = (ph.get(q) or {})
            if a.get("answer") not in VOCABULARY["yes_no_unsure"]:
                raise StepError("Answer every processing-history question (yes, no or not sure).")
            d["processing_history"][q] = {"answer": a["answer"], "note": (a.get("note") or "").strip()}
        d["software_and_version"] = (ph.get("software_and_version") or "").strip()
        d["history_notes"] = (ph.get("notes") or "").strip()

    if "metadata" in decision:
        apply_metadata_decision(s, d, decision["metadata"])

    _check_step(s, d, step)
    _assign_block_roles(d, s)
    refresh_sample_types(s, d)
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
        dup = long_duplicates(s, d)
        if dup:
            d["long_duplicates"] = dup
    if step == "values":
        per = Counter(it.get("omics_type") or "unknown" for it in d["groups"].values()
                      if it["role"] == "value" and it.get("block_role") == "primary")
        if not per:
            raise StepError("Mark at least one value block as primary.")
        too_many = [o for o, n in per.items() if n > 1]
        if too_many:
            raise StepError(f"Only one primary block per omics type (more than one for: {', '.join(too_many)}).")
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
    for key in ("layout", "omics_type", "source_software"):
        if before[key]["value"] != after[key]["value"]:
            out.append({"field": key, "from": before[key]["value"], "to": after[key]["value"],
                        "proposal": after[key].get("proposed"), "provenance": provenance(after[key], FACT_FIELDS)})
    return out


def long_duplicates(s, d):
    """(feature, sample) pairs occurring more than once in a long table."""
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


# ---------------------------------------------------------------- reconsider

def reconsider(s, gid, hint, on_progress=None):
    d = s.draft
    if gid not in d["groups"]:
        raise StepError(f"Unknown group '{gid}'.")
    ok, why = llm.available()
    if not (ok and d["ai"]["enabled"]):
        raise StepError("The AI is off or unavailable: " + (why or "turn it on to ask it to reconsider."))
    fixed = {"layout": d["layout"]["value"], "omics_type": d["omics_type"]["value"],
             "confirmed_groups": {g: {k: it.get(k) for k in ("role", "kind", "measurement_type", "scale")}
                                  for g, it in d["groups"].items() if g != gid and it["role"] != UNRESOLVED},
             "user_hint_for_this_group": (hint or "").strip()[:500],
             "instruction": f"Reconsider group {gid} only, taking the user's hint into account."}
    prop, meta, digests = ai.propose(s.filename, s.sha, s.groups, s.hints, s.cols, fixed=fixed,
                                     group_ids={gid}, log=s.log, on_progress=on_progress)
    new = prop["groups"].get(gid)
    if meta.get("error") or new is None:
        raise StepError("The AI could not answer: " + (meta.get("error") or "no label returned"))
    old = d["groups"][gid]
    new = _blank(new)
    new["hint"] = old.get("hint")
    new["keep"] = old.get("keep", True)
    if new["role"] == "value":
        new["block_role"] = old.get("block_role") or "auxiliary"
        new["omics_type"] = new.get("omics_type") or old.get("omics_type")
    _snap(new, GROUP_FIELDS)
    d["groups"][gid] = new
    s.log("reconsider", {"group_id": gid, "user_hint": hint, "old": {k: old.get(k) for k in GROUP_FIELDS},
                         "new": {k: new.get(k) for k in GROUP_FIELDS}, "validation": new["validation"]})
    s.save()
    return {"draft": public_draft(s), "digest": digests[0] if digests else None}


# ---------------------------------------------------------------- sample metadata file (samples in columns)

def _norm_id(x):
    x = x.strip().lower()
    x = re.sub(r"[\s_\-.]+", "", x)
    return re.sub(r"(?<![0-9])0+(?=[0-9])", "", x)


def upload_metadata(s, filename, raw):
    t = parse_bytes(sanitize_filename(filename), raw)
    cols = Columns(t)
    groups = [g for g in build_groups(cols)]
    data_ids = sample_ids(s, s.draft)
    best, best_hits = 0, -1
    for i in range(len(t["header"])):
        vals = {cell(r, i).strip() for r in t["rows"]}
        hits = len(vals & set(data_ids))
        if hits > best_hits:
            best, best_hits = i, hits
    s.metadata_table = {"table": t, "cols": cols, "groups": groups}
    meta = {"filename": sanitize_filename(filename), "id_column": t["header"][best],
            "columns": [{"column": cols.labels[i], "index": i, "hint": group_hint(
                {"kind": "single_column", "type": cols.digests[i]["type"], "profile": cols.digests[i],
                 "n_columns": 1}), "role": "sample_metadata" if i != best else "sample_id",
                "kind": None, "detail": None, "keep": True} for i in range(len(t["header"]))],
            "accepted_near_misses": [], "skipped": False}
    _guess_metadata_kinds(meta, cols)
    meta["report"] = match_report(data_ids, [cell(r, best).strip() for r in t["rows"]])
    s.draft["metadata"] = meta
    s.log("metadata_upload", {"filename": meta["filename"], "id_column": meta["id_column"],
                              "report": {k: v for k, v in meta["report"].items() if k != "matched"}})
    s.save()
    return public_draft(s)


def _guess_metadata_kinds(meta, cols):
    from .mock_llm import _TEXT_RULES, _NUM_RULES, _first
    for c in meta["columns"]:
        if c["role"] == "sample_id":
            continue
        d = cols.digests[c["index"]]
        rule = _first(c["column"], _NUM_RULES if d["type"] == "numeric" else _TEXT_RULES)
        if rule and rule[0] == "sample_metadata":
            c["kind"] = rule[1]
            c["source"] = "computed"
        else:
            c["kind"] = "covariate_numeric" if d["type"] == "numeric" else None
            c["source"] = "computed" if c["kind"] else "none"
        if c["kind"] == "timepoint":
            c["detail"] = timepoint_detail(d)
        c["proposed"] = {"kind": c["kind"], "role": c["role"]}


def match_report(data_ids, meta_ids):
    ds, ms = set(data_ids), set(i for i in meta_ids if i)
    matched = sorted(ds & ms)
    only_data = [i for i in data_ids if i not in ms]
    only_meta = [i for i in meta_ids if i and i not in ds]
    norm_meta = {}
    for m in only_meta:
        norm_meta.setdefault(_norm_id(m), []).append(m)
    near = []
    for dd in only_data:
        for m in norm_meta.get(_norm_id(dd), []):
            near.append({"data_id": dd, "metadata_id": m,
                         "reason": "differs only in case, spaces/underscores or leading zeros"})
    return {"matched": matched, "n_matched": len(matched), "only_in_data": only_data,
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
        if c and c["role"] != "sample_id":
            c["kind"] = ed.get("kind")
            c["keep"] = bool(ed.get("keep", True))
            if c["kind"] not in VOCABULARY["sample_metadata_kind"]:
                raise StepError(f"Choose a kind for metadata column '{c['column']}'.")


# ---------------------------------------------------------------- public view

def public_draft(s):
    """Draft + derived facts for the frontend (provenance, sample ids, groups)."""
    d = s.draft
    out = copy.deepcopy(d)
    for gid, it in out["groups"].items():
        it["provenance"] = provenance(it, GROUP_FIELDS)
    for key in ("layout", "omics_type", "source_software"):
        out[key]["provenance"] = provenance(out[key], FACT_FIELDS)
    out["feature_identity"]["provenance"] = provenance(out["feature_identity"], FI_FIELDS)
    ids = sample_ids(s, d)
    dup = [k for k, n in Counter(ids).items() if n > 1]
    out["samples"] = {"ids": ids, "n": len(ids), "duplicates": dup}
    out["blocks_samples"] = {gid: sample_ids(s, d, gid) for gid, it in d["groups"].items()
                             if it["role"] == "value" and layout_of(d) == "samples_in_columns"}
    for sid, st in out["sample_types"].items():
        st["provenance"] = provenance(st, ("type",))
    out["unresolved"] = unresolved_items(s, d)
    out["flags"] = {gid: flag_counts(s, gid) for gid, it in d["groups"].items()
                    if (it.get("kind") or "").startswith("flag_")}
    return out


def flag_counts(s, gid):
    g = s.groups_by_id[gid]
    i = g["indices"][0]
    n = sum(1 for r in s.table["rows"] if not is_missing(cell(r, i)) and cell(r, i).strip() not in ("0", "false", "False"))
    return n


def unresolved_items(s, d):
    out = []
    if d["layout"]["value"] not in VOCABULARY["layout"]:
        out.append({"step": "layout", "what": "Layout is not decided."})
    for gid, it in d["groups"].items():
        if it["role"] == UNRESOLVED:
            out.append({"step": step_for_group(s, d, gid), "group_id": gid,
                        "what": f"Role of {', '.join(s.groups_by_id[gid]['columns'][:2])}"
                                + (" ..." if s.groups_by_id[gid]["n_columns"] > 2 else "") + " is unresolved."})
        elif it["role"] in ("feature_annotation", "sample_metadata") and not it.get("kind"):
            out.append({"step": step_for_group(s, d, gid), "group_id": gid,
                        "what": f"Kind of '{s.groups_by_id[gid]['columns'][0]}' is not chosen."})
    lay = layout_of(d)
    if lay in ("samples_in_columns", "long") and not d["feature_identity"]["group_ids"]:
        out.append({"step": "feature_id", "what": "Feature identity is not chosen."})
    if lay in ("samples_in_rows", "long") and not d["sample_id_group"]["value"]:
        out.append({"step": "samples", "what": "Sample ID column is not chosen."})
    if d.get("long_duplicates") and lay == "long":
        out.append({"step": "feature_id", "what": d["long_duplicates"]["message"]})
    if not primary_blocks(s, d):
        out.append({"step": "values", "what": "No primary value block."})
    for q, _ in HISTORY_QUESTIONS:
        if not d["processing_history"][q]["answer"]:
            out.append({"step": "history", "what": "Processing-history questions are not all answered."})
            break
    for st in STEPS[:-1]:
        if d["steps"][st] == "pending":
            out.append({"step": st, "what": f"Step '{st.replace('_', ' ')}' is not confirmed yet."})
    return out


def step_for_group(s, d, gid):
    it = d["groups"][gid]
    g = s.groups_by_id[gid]
    lay = layout_of(d)
    role = it["role"]
    if role == "feature_id":
        return "feature_id"
    if role == "feature_annotation":
        return "annotations"
    if role == "value":
        return "values"
    if role in ("sample_id", "sample_metadata"):
        return "sample_info" if role == "sample_metadata" else "samples"
    if role == "ignore":
        return "values" if g["type"] == "numeric" else ("sample_info" if lay == "samples_in_rows" else "annotations")
    # unresolved: numeric blocks -> values; numeric singles -> sample info in rows layout, else annotations
    if g["kind"] == "numeric_block":
        return "values"
    if lay == "samples_in_rows":
        return "values" if g["type"] == "numeric" else "sample_info"
    return "annotations"
