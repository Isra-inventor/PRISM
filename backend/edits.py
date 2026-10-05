"""The single edit layer (v2.4 §3): every change to a session's schema goes through
apply_edit(session, op, args, actor).

- actor is 'user' (a wizard control), 'ai_patch' (an AI patch the user clicked) or
  'question_option' (an option of a queued question the user chose).
- Each op validates first and only then changes the draft; a failing op leaves the
  draft exactly as it was (the whole edit is rolled back).
- Provenance: an edit by the user gives user_set; an AI patch or question option
  gives ai_proposed_confirmed (ai_proposed_corrected when the user edited it first).
- Every top-level edit is pushed onto a per-session undo stack (UNDO_MAX entries) as
  a compact diff (changed draft keys, changed group / sample entries, the old group
  structure when it changed) and logged with before and after.
- Nothing is ever deleted from the source files: 'exclude' only leaves a column out
  of the outputs, recorded with a reason, who did it and when.
- Finalized datasets are immutable.

Ops are registered with @op(name, ai=...). Only ops marked ai=True can be run by an
AI patch or a question option the AI wrote (the closed set of v2.4 §4.3); the
others are wizard controls (layout, sample rules, processing history, ...).
"""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager

from . import ai
from .errors import StepError
from .profiling import make_group
from .schema import UNRESOLVED, VOCABULARY
from .session_log import now_iso
from .validation import timepoint_detail, validate_group

ACTORS = ("user", "ai_patch", "question_option")
UNDO_MAX = 150
_MISSING = {"__missing__": True}


class EditError(StepError):
    pass


OPS = {}


def op(name, ai=False):
    def deco(fn):
        OPS[name] = {"fn": fn, "ai": ai}
        return fn
    return deco


class Ctx:
    """What an op sees: the session, the draft, who acts, and where to report."""

    def __init__(self, s, actor, reason=None, corrected=False, step=None, extra=None, tx=None):
        self.s, self.d, self.actor = s, s.draft, actor
        self.reason, self.corrected, self.step = reason, corrected, step
        self.extra = extra or {}
        self.tx = tx
        self.warnings = []
        self.columns = []          # affected columns as (file, column)
        self.touched = {}          # gid -> fields before
        self.summary = ""

    def touch(self, gid):
        if gid not in self.touched and gid in self.d["groups"]:
            self.touched[gid] = {f: self.d["groups"][gid].get(f) for f in wf.GROUP_FIELDS}
            g = self.s.groups_by_id.get(gid)
            if g:
                self.columns.extend(("main", c) for c in g["columns"])


# ---------------------------------------------------------------- recording, rollback, undo

def _structure(groups):
    return [{k: g.get(k) for k in ("group_id", "indices", "origin", "split_from", "merged_from")} for g in groups]


def _same_structure(a, b):
    return len(a) == len(b) and all(x["group_id"] == y["group_id"] and x["indices"] == y["indices"]
                                    for x, y in zip(a, b))


def _json_copy(x):
    return json.loads(json.dumps(x))


@contextmanager
def recording(s, op_name, actor="user", summary="", args=None):
    """One undoable, atomic edit. Nested calls join the outer edit."""
    if getattr(s, "_tx", None) is not None:
        yield s._tx
        return
    if s.draft is None:
        raise EditError("Run the proposal first.")
    if s.draft.get("finalized"):
        raise EditError("This dataset is finalized; its schema can no longer change.")
    snap = {"draft": _json_copy(s.draft), "groups": list(s.groups)}
    tx = {"op": op_name, "actor": actor, "summary": summary, "args": args, "columns": [], "warnings": []}
    s._tx = tx
    try:
        yield tx
        wf.after_edit(s)
    except BaseException:
        s.draft, s.groups = snap["draft"], snap["groups"]
        raise
    finally:
        s._tx = None
    _commit(s, snap, tx)


def _diff(before, after):
    keys, per = {}, {}
    for k in set(before) | set(after):
        b, a = before.get(k, _MISSING), after.get(k, _MISSING)
        if k in ("groups", "samples") and isinstance(b, dict) and isinstance(a, dict):
            ch = {x: b.get(x, _MISSING) for x in set(b) | set(a) if b.get(x, _MISSING) != a.get(x, _MISSING)}
            if ch:
                per[k] = ch
        elif b != a:
            keys[k] = b
    return keys, per


def _commit(s, snap, tx):
    keys, per = _diff(snap["draft"], s.draft)
    old_struct = _structure(snap["groups"])
    struct_changed = not _same_structure(old_struct, _structure(s.groups))
    if not keys and not per and not struct_changed:
        return None
    cols = list(dict.fromkeys(tx["columns"]))
    entry = {"edit_id": uuid.uuid4().hex[:10], "at": now_iso(), "actor": tx["actor"], "op": tx["op"],
             "summary": tx["summary"] or tx["op"].replace("_", " "), "n_columns": len(cols),
             "columns": [{"file": f, "column": c} for f, c in cols[:50]],
             "restore": {"keys": keys, "per": per, "structure": old_struct if struct_changed else None}}
    s.undo.append(entry)
    del s.undo[:-UNDO_MAX]
    s.undo_dirty = True
    changes = {}
    for gid, old in per.get("groups", {}).items():
        new = s.draft["groups"].get(gid, _MISSING)
        if old is _MISSING or new is _MISSING or old == _MISSING or new == _MISSING:
            changes[gid] = {"group": "added" if old == _MISSING else "removed" if new is _MISSING else "replaced"}
            continue
        ch = {f: [old.get(f), new.get(f)] for f in wf.GROUP_FIELDS if old.get(f) != new.get(f)}
        if ch:
            changes[gid] = ch
        if len(changes) >= 300:
            break
    event = "edit_applied" if tx["actor"] == "user" else "patch_applied"
    s.log(event, {"edit_id": entry["edit_id"], "op": tx["op"], "actor": tx["actor"], "summary": entry["summary"],
                  "args": tx.get("args"), "n_columns": len(cols), "columns": [c for _, c in cols[:100]],
                  "group_changes": changes, "changed_keys": sorted(keys), "structure_changed": struct_changed,
                  "warnings": tx["warnings"]})
    return entry


def _rebuild_groups(s, structure):
    cur = {g["group_id"]: g for g in s.groups}
    out = []
    for st in structure:
        g = cur.get(st["group_id"])
        if g is None or g["indices"] != sorted(st["indices"]):
            g = make_group(s.cols, st["group_id"], st["indices"], st.get("origin") or "restored",
                           split_from=st.get("split_from"), merged_from=st.get("merged_from"))
        out.append(g)
    s.groups = out


def undo(s, edit_id=None):
    """Undo the latest edit, or every edit back to and including edit_id."""
    if s.draft is None:
        raise EditError("Nothing to undo.")
    if s.draft.get("finalized"):
        raise EditError("This dataset is finalized; its schema can no longer change.")
    if not s.undo:
        raise EditError("Nothing to undo.")
    if edit_id and edit_id not in {e["edit_id"] for e in s.undo}:
        raise EditError("That change can no longer be undone.")
    d = s.draft
    undone = []
    while s.undo:
        e = s.undo.pop()
        r = e["restore"]
        for k, v in r["keys"].items():
            if v == _MISSING:
                d.pop(k, None)
            else:
                d[k] = v
        for k, entries in r["per"].items():
            for x, v in entries.items():
                if v == _MISSING:
                    d[k].pop(x, None)
                else:
                    d[k][x] = v
        if r["structure"] is not None:
            _rebuild_groups(s, r["structure"])
        undone.append(e)
        s.log("patch_undone", {"edit_id": e["edit_id"], "op": e["op"], "actor": e["actor"], "summary": e["summary"]})
        if edit_id is None or e["edit_id"] == edit_id:
            break
    d["groups"] = {g["group_id"]: d["groups"][g["group_id"]] for g in s.groups if g["group_id"] in d["groups"]}
    s.undo_dirty = True
    s.save()
    return [{"edit_id": e["edit_id"], "summary": e["summary"]} for e in undone]


def changes(s, limit=100):
    """The 'Changes' list: latest first, without the restore data."""
    return [{k: e[k] for k in ("edit_id", "at", "actor", "op", "summary", "n_columns")}
            for e in reversed(s.undo[-limit:])]


# ---------------------------------------------------------------- entry point

def apply_edit(s, op_name, args=None, actor="user", reason=None, corrected=False, step=None, extra=None,
               summary=None):
    """Validate and apply one op. Returns {before, after, affected_columns, warnings, summary, result}."""
    if actor not in ACTORS:
        raise EditError(f"Unknown actor '{actor}'.")
    spec = OPS.get(op_name)
    if spec is None:
        raise EditError(f"Unknown operation '{op_name}'.")
    if actor != "user" and not spec["ai"]:
        raise EditError(f"'{op_name}' can only be done by you in the wizard, not by an AI patch.")
    args = dict(args or {})
    with recording(s, op_name, actor, summary or "", _loggable(args)) as tx:
        ctx = Ctx(s, actor, reason, corrected, step, extra, tx)
        result = spec["fn"](ctx, args)
        if summary is None and ctx.summary and tx["op"] == op_name and not tx["summary"]:
            tx["summary"] = ctx.summary
        tx["columns"].extend(ctx.columns)
        tx["warnings"].extend(ctx.warnings)
    after = {gid: {f: s.draft["groups"][gid].get(f) for f in wf.GROUP_FIELDS}
             for gid in ctx.touched if gid in s.draft["groups"]}
    return {"before": ctx.touched, "after": after, "affected_columns": [c for _, c in dict.fromkeys(ctx.columns)],
            "warnings": ctx.warnings, "summary": ctx.summary, "result": result}


def _loggable(args):
    return {k: v for k, v in args.items() if not callable(v)}


# ---------------------------------------------------------------- group fields

def excluded_record(ctx, default_reason):
    return {"reason": ctx.reason or default_reason, "by": ctx.actor, "at": now_iso()}


def _clean(text, limit=300):
    return (str(text) if text is not None else "").strip()[:limit]


def set_group_fields(ctx, gid, fields):
    """Validate, then set group fields (role, label, keep, ...). Returns True when the role changed."""
    s, d = ctx.s, ctx.d
    if gid not in d["groups"]:
        raise EditError(f"Unknown group '{gid}'.")
    cur = d["groups"][gid]
    it = _json_copy(cur)
    for f in wf.GROUP_FIELDS + ("family", "flagged_values"):
        if f not in fields:
            continue
        val = fields[f]
        if f in ("assay_label", "detail", "family"):
            val = _clean(val, 120 if f == "assay_label" else 300) or None
        elif f == "label":
            val = _clean(val)
        elif f in ("keep", "marks_rows_as_suspect"):
            val = bool(val)
        elif f == "role" and val not in VOCABULARY["column_role"]:
            raise EditError(f"'{val}' is not an allowed role.")
        elif f == "audit_kind" and val is not None and val not in VOCABULARY["audit_kind"]:
            raise EditError(f"'{val}' is not an allowed audit kind.")
        it[f] = val
    ai.normalize_item(it)
    g = s.groups_by_id[gid]
    if it["role"] == "value" and not it.get("assay_label"):
        it["assay_label"] = d["assays"][0]["assay_label"]
    if it.get("marks_rows_as_suspect") and it.get("flag_values") is None:
        it["flag_values"] = wf.flag_values(s, gid)
    if it["role"] == "sample_metadata" and it.get("audit_kind") == "timepoint" and not it.get("detail") \
            and g["n_columns"] == 1:
        it["detail"] = timepoint_detail(s.cols.digests[g["indices"][0]])
    it["validation"] = validate_group(it, g, s.cols, wf.layout_of(d))
    if it["validation"]["status"] == "contradicted":
        raise EditError(f"{', '.join(g['columns'][:2])}: " + " ".join(it["validation"]["messages"]))
    changed = [f for f in wf.GROUP_FIELDS + ("family", "flagged_values") if it.get(f) != cur.get(f)]
    if not changed:
        return False
    ctx.touch(gid)
    excluded = it["role"] == "ignore" or not it.get("keep", True)
    if excluded and not cur.get("excluded"):
        it["excluded"] = excluded_record(ctx, "role set to 'ignore'" if it["role"] == "ignore" else "left out of the outputs")
    elif not excluded:
        it.pop("excluded", None)
    it["set_by"], it["set_corrected"] = ctx.actor, bool(ctx.corrected)
    if it["role"] == "value" and it["assay_label"] not in [a["assay_label"] for a in d["assays"]]:
        d["assays"].append(wf._snap({"assay_label": it["assay_label"], "omics_type": "unknown",
                                     "source_software": "unknown", "in_supported_scope": "unsure", "scope_reason": "",
                                     "source": "user", "confidence": 1.0,
                                     "evidence": "Added by you." if ctx.actor == "user" else "Added by an AI patch."},
                                    wf.ASSAY_FIELDS))
    old_step = wf.step_for_group(s, d, gid)
    d["groups"][gid] = it
    role_changed = it["role"] != cur["role"]
    if ctx.step is None:   # an edit outside a step confirmation reopens the steps it touches
        for st in {old_step, wf.step_for_group(s, d, gid)}:
            if d["steps"].get(st) == "confirmed":
                d["steps"][st] = "pending"
    return role_changed


@op("edit_group")
def _edit_group(ctx, a):
    """The wizard's inline editor: several fields of one group at once."""
    gid = a.get("group_id")
    changed = set_group_fields(ctx, gid, {k: v for k, v in (a.get("fields") or {}).items()})
    ctx.summary = f"Edited {', '.join(ctx.s.groups_by_id[gid]['columns'][:2])}" if gid in ctx.s.groups_by_id else ""
    return {"role_changed": changed}


from . import workflow as wf  # noqa: E402  (last: workflow registers its ops on import)
