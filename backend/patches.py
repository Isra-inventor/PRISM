"""AI patches (v2.4 §4): the AI acts on the schema only through these.

A patch is {op, target: {selector, except_columns}, args, reason, consequences}.
Code resolves the selector to a concrete column list (a name the AI invents is
rejected), checks the op against the closed set and the invariants, and previews
it on a copy of the draft (the diff card). Nothing is applied until you click;
applying runs the patch through the edit layer (actor 'ai_patch', or
'question_option' for an option of a queued question), so it is validated,
logged and undoable like any other edit. `reason` and `consequences` are text
for you only; code never trusts them.
"""

from __future__ import annotations

import fnmatch
import json
import uuid

from . import edits
from .errors import StepError
from .schema import UNRESOLVED, VOCABULARY

LARGE = 25          # an op touching more columns than this needs an explicit click showing the count
AI_OPS = ["set_keep", "set_role", "set_audit_kind", "set_label", "set_block_keep", "merge_groups", "split_group",
          "merge_assays", "derive_feature_annotation", "set_design", "set_sample_label", "set_join_key",
          "set_flag_values"]
COLUMN_OPS = {"set_keep", "set_role", "set_audit_kind", "set_label", "set_block_keep", "merge_groups",
              "split_group", "set_flag_values"}
SAMPLE_OPS = {"set_sample_label"}
SELECTOR_KEYS = ("columns", "group_id", "group_ids", "role", "audit_kind", "file", "name_contains",
                 "name_starts_with", "name_ends_with", "samples")
OP_TEXT = {"set_keep": "keep / exclude", "set_role": "set role", "set_audit_kind": "set audit kind",
           "set_label": "set label", "set_block_keep": "keep / exclude block", "merge_groups": "merge groups",
           "split_group": "take columns out of their group", "merge_assays": "merge assays",
           "derive_feature_annotation": "derive annotation from feature names", "set_design": "set study design",
           "set_sample_label": "label samples", "set_join_key": "metadata join key",
           "set_flag_values": "flag rows by value"}


class PatchError(StepError):
    pass


# ---------------------------------------------------------------- selectors

def _meta_cols(d, use):
    meta = d.get("metadata") or {}
    if not use or meta.get("skipped"):
        return []
    return meta.get("columns", [])


def resolve(s, d, op, target):
    """Selector -> {"main": [column index], "metadata": [names], "samples": [ids]}. Raises PatchError."""
    target = target or {}
    sel = dict(target.get("selector") or {})
    if op in SAMPLE_OPS:
        pats = sel.get("samples") or []
        if not pats:
            raise PatchError("Name the samples (names or patterns such as 'QC_*').")
        hit = [x for x in d["samples"] if any(x == p or fnmatch.fnmatchcase(x, p) for p in pats)]
        if not hit:
            raise PatchError(f"No sample matches {', '.join(pats[:5])}.")
        return {"main": [], "metadata": [], "samples": hit}
    if not any(sel.get(k) not in (None, "", []) for k in SELECTOR_KEYS):
        raise PatchError("The target selects nothing in particular; name columns, a group, a role or a name part.")
    f = sel.get("file")
    if f not in (None, "", "main", "metadata"):
        raise PatchError(f"file must be 'main' or 'metadata', not '{f}'.")
    labels = s.cols.labels
    main = list(range(len(labels))) if f in (None, "", "main") else []
    metas = list(_meta_cols(d, f in (None, "", "metadata")))
    all_main = set(labels)
    all_meta = {c["column"] for c in _meta_cols(d, True)}

    def known(names, what):
        bad = [n for n in names if n not in all_main and n not in all_meta]
        if bad:
            raise PatchError(f"{what} {', '.join(repr(b) for b in bad[:5])} do(es) not exist in the data.")

    if sel.get("columns"):
        names = set(sel["columns"])
        known(sel["columns"], "Column(s)")
        main = [i for i in main if labels[i] in names]
        metas = [c for c in metas if c["column"] in names]
    gids = [g for g in ([sel["group_id"]] if sel.get("group_id") else []) + list(sel.get("group_ids") or [])]
    if gids:
        bad = [g for g in gids if g not in d["groups"]]
        if bad:
            raise PatchError(f"Group(s) {', '.join(bad)} do(es) not exist.")
        idx = {i for g in gids for i in s.groups_by_id[g]["indices"]}
        main = [i for i in main if i in idx]
        metas = []
    owner = {i: g["group_id"] for g in s.groups for i in g["indices"]}
    if sel.get("role"):
        role = sel["role"]
        if role not in VOCABULARY["column_role"]:
            raise PatchError(f"'{role}' is not a role.")
        main = [i for i in main if d["groups"][owner[i]]["role"] == role]
        metas = [c for c in metas if c["role"] == role]
    if sel.get("audit_kind"):
        main = [i for i in main if d["groups"][owner[i]].get("audit_kind") == sel["audit_kind"]]
        metas = [c for c in metas if c.get("audit_kind") == sel["audit_kind"]]
    for key, test in (("name_contains", lambda n, x: x in n), ("name_starts_with", lambda n, x: n.startswith(x)),
                      ("name_ends_with", lambda n, x: n.endswith(x))):
        x = sel.get(key)
        if x:
            x = x.lower()
            main = [i for i in main if test(labels[i].lower(), x)]
            metas = [c for c in metas if test(c["column"].lower(), x)]
    exc = target.get("except_columns") or []
    if exc:
        known(exc, "except_columns")
        main = [i for i in main if labels[i] not in exc]
        metas = [c for c in metas if c["column"] not in exc]
    if not main and not metas:
        raise PatchError("The target matches no column.")
    return {"main": main, "metadata": [c["column"] for c in metas], "samples": []}


def _owners(s, idx):
    out = {}
    pos = {i: g["group_id"] for g in s.groups for i in g["indices"]}
    for i in idx:
        out.setdefault(pos[i], []).append(i)
    return out


def _isolate(s, d, idx, ctx_warn):
    """Group ids covering exactly these columns: a group only partly selected is split
    so the change applies to the selected columns only (said in a warning)."""
    from . import workflow as wf
    out = []
    for gid, cols in _owners(s, idx).items():
        g = s.groups_by_id[gid]
        if len(cols) == g["n_columns"]:
            out.append(gid)
            continue
        item = json.loads(json.dumps(d["groups"][gid]))
        pg = {"rest": {"indices": [i for i in g["indices"] if i not in cols], "item": json.loads(json.dumps(item)),
                       "origin": g["origin"], "keep_id": gid},
              "sel": {"indices": cols, "item": item, "origin": "split_patch", "split_from": gid}}
        new = wf._replace_groups(s, d, [gid], pg, {"patch_isolate": gid})
        out.extend(n for n in new if n != gid)
        ctx_warn.append(f"{len(cols)} of the {g['n_columns']} columns of '{d['groups'][gid].get('label') or g['columns'][0]}' "
                        "were taken out of their group so the change applies to them only.")
    return out


# ---------------------------------------------------------------- running a patch

def _run(s, patch, actor, corrected=False, args_override=None):
    """Apply one patch inside the current edit (caller holds the recording)."""
    d = s.draft
    op, args = patch["op"], dict(args_override if args_override is not None else patch.get("args") or {})
    tx = s._tx
    if op not in AI_OPS:
        raise PatchError(f"'{op}' is not an allowed operation.")
    if op in COLUMN_OPS or op in SAMPLE_OPS:
        res = resolve(s, d, op, patch.get("target"))
    else:
        res = {"main": [], "metadata": [], "samples": []}
    kw = {"reason": patch.get("reason"), "corrected": corrected}
    if op in COLUMN_OPS:
        if op == "merge_groups":
            gids = list(_owners(s, res["main"]))
            if res["metadata"]:
                raise PatchError("Metadata columns are not grouped.")
            return edits.apply_edit(s, op, dict(args, group_ids=gids), actor, **kw)
        if op == "split_group":
            out = None
            for gid, cols in _owners(s, res["main"]).items():
                out = edits.apply_edit(s, op, {"group_id": gid, "columns": [s.cols.labels[i] for i in cols]}, actor, **kw)
            return out
        gids = _isolate(s, d, res["main"], tx["warnings"])
        return edits.apply_edit(s, op, dict(args, group_ids=gids, metadata_columns=res["metadata"]), actor, **kw)
    if op in SAMPLE_OPS:
        return edits.apply_edit(s, op, dict(args, sample_ids=res["samples"]), actor, **kw)
    return edits.apply_edit(s, op, args, actor, **kw)


def describe_target(s, d, patch):
    try:
        res = resolve(s, d, patch["op"], patch.get("target")) if patch["op"] in COLUMN_OPS | SAMPLE_OPS else None
    except PatchError:
        return None
    if res is None:
        return None
    cols = [("main", s.cols.labels[i]) for i in res["main"]] + [("metadata", c) for c in res["metadata"]]
    return {"n_columns": len(cols), "columns": [c for _, c in cols][:500],
            "files": sorted({f for f, _ in cols}), "n_samples": len(res["samples"]), "samples": res["samples"][:200]}


def _observe(s):
    def observe(before, tx):
        from . import accounting
        after = s.draft
        bg, ag = before["groups"], after["groups"]
        changed = []
        for gid, it in ag.items():
            b = bg.get(gid)
            if b is None:
                continue
            ch = {f: [b.get(f), it.get(f)] for f in ("role", "label", "audit_kind", "keep", "assay_label",
                                                       "flagged_values", "family") if b.get(f) != it.get(f)}
            if ch:
                g = s.groups_by_id[gid]
                changed.append({"group_id": gid, "columns": g["columns"][:3], "n_columns": g["n_columns"], "changes": ch})
        meta_changed = []
        bm = {c["column"]: c for c in (before.get("metadata") or {}).get("columns", [])}
        for c in (after.get("metadata") or {}).get("columns", []):
            b = bm.get(c["column"])
            if b:
                ch = {f: [b.get(f), c.get(f)] for f in ("role", "label", "audit_kind", "keep") if b.get(f) != c.get(f)}
                if ch:
                    meta_changed.append({"column": c["column"], "changes": ch})
        out = {"n_groups_changed": len(changed), "groups": changed[:6], "metadata": meta_changed[:6],
               "n_metadata_changed": len(meta_changed),
               "n_groups_before": len(bg), "n_groups_after": len(ag),
               "warnings": list(tx["warnings"])}
        if [a["assay_label"] for a in before["assays"]] != [a["assay_label"] for a in after["assays"]]:
            out["assays"] = [[a["assay_label"] for a in before["assays"]], [a["assay_label"] for a in after["assays"]]]
        ns = sum(1 for k, v in after["samples"].items() if before["samples"].get(k) != v)
        if ns:
            out["n_samples_changed"] = ns
        if before.get("derived_feature_annotations") != after.get("derived_feature_annotations"):
            out["derived"] = [x["name"] for x in after.get("derived_feature_annotations", [])
                              if x not in (before.get("derived_feature_annotations") or [])]
        out["ledger_after"] = {f: {k: v for k, v in x.items() if k != "problems"}
                               for f, x in accounting.column_ledger(s, after).items()}
        return out
    return observe


def prepare(s, raw, source="chat", message_id=None):
    """Validate an AI patch and preview it on a copy. Returns the stored patch
    (status 'pending', or 'rejected' with the reason). Nothing is applied."""
    d = s.draft
    patch = {"patch_id": "p" + uuid.uuid4().hex[:8], "ai_patch_id": raw.get("patch_id"), "op": raw.get("op"),
             "target": raw.get("target") or {}, "args": raw.get("args") or {}, "reason": raw.get("reason") or "",
             "consequences": list(raw.get("consequences") or []), "source": source, "message_id": message_id,
             "status": "pending", "problems": [], "op_text": OP_TEXT.get(raw.get("op"), raw.get("op"))}
    if "processing_history" in json.dumps(raw).lower() or patch["op"] == "set_processing_history":
        return _reject(s, d, patch, "Processing history is answered by you only; the AI cannot answer it.")
    if patch["op"] not in AI_OPS:
        return _reject(s, d, patch, f"'{patch['op']}' is not an allowed operation.")
    patch["resolved"] = describe_target(s, d, patch)
    try:
        r, obs = edits.trial(s, lambda: _run(s, patch, "ai_patch"), _observe(s))
        patch["preview"] = obs
        patch["warnings"] = obs["warnings"]
        if patch["op"] == "derive_feature_annotation" and r:
            patch["derivation"] = r.get("result")
    except StepError as e:
        return _reject(s, d, patch, str(e))
    n = (patch["resolved"] or {}).get("n_columns") or 0
    patch["n_columns"] = n
    patch["large"] = n > LARGE
    _store(d, patch)
    s.log("ai_patch_proposed", {k: patch.get(k) for k in ("patch_id", "op", "target", "args", "reason", "consequences",
                                                         "source", "n_columns", "large", "warnings")})
    return patch


def _reject(s, d, patch, why):
    patch.update(status="rejected", problems=[why])
    _store(d, patch)
    s.log("patch_rejected", {"patch_id": patch["patch_id"], "op": patch["op"], "target": patch["target"],
                             "args": patch["args"], "reason": why, "when": "proposed"})
    return patch


def _store(d, patch):
    ps = d.setdefault("patches", [])
    ps.append(patch)
    del ps[:-300]


def find(d, pid):
    return next((p for p in d.get("patches", []) if p["patch_id"] == pid), None)


def apply(s, patch_ids, confirm_large=(), overrides=None, actor="ai_patch"):
    """Apply pending patches, each as its own undoable edit. Returns per-patch results."""
    overrides = overrides or {}
    results = []
    for pid in patch_ids:
        d = s.draft
        p = find(d, pid)
        if p is None:
            results.append({"patch_id": pid, "status": "rejected", "reason": "Unknown patch."})
            continue
        if p["status"] != "pending" and p["status"] != "held":
            results.append({"patch_id": pid, "status": p["status"], "reason": "Not pending."})
            continue
        res = describe_target(s, d, p)
        n = (res or {}).get("n_columns") or 0
        if n > LARGE and pid not in confirm_large:
            why = f"This touches {n} columns: confirm the count to apply it."
            p.update(status="held", problems=[why], n_columns=n, large=True)
            results.append({"patch_id": pid, "status": "held", "reason": why, "n_columns": n})
            continue
        args = overrides.get(pid)
        corrected = args is not None and args != p.get("args")
        try:
            with edits.recording(s, p["op"], actor, f"AI: {_summary(s, p)}", {"patch_id": pid, "op": p["op"]}) as tx:
                tx["edit_id"] = eid = "e" + uuid.uuid4().hex[:9]
                _run(s, p, actor, corrected, args)
                q = find(s.draft, pid)
                q.update(status="applied", edit_id=eid, applied_args=args if corrected else None, problems=[],
                         warnings=list(tx["warnings"]))
            results.append({"patch_id": pid, "status": "applied", "edit_id": eid, "warnings": q["warnings"]})
        except StepError as e:
            q = find(s.draft, pid)
            q.update(status="rejected", problems=[str(e)])
            s.log("patch_rejected", {"patch_id": pid, "op": p["op"], "reason": str(e), "when": "applied"})
            results.append({"patch_id": pid, "status": "rejected", "reason": str(e)})
    s.save()
    return results


def dismiss(s, patch_ids):
    d = s.draft
    for pid in patch_ids:
        p = find(d, pid)
        if p and p["status"] in ("pending", "held"):
            p["status"] = "dismissed"
            s.log("patch_rejected", {"patch_id": pid, "op": p["op"], "reason": "dismissed by you", "when": "dismissed"})
    s.save()


def _summary(s, p):
    n = (p.get("resolved") or {}).get("n_columns")
    a = p.get("args") or {}
    what = {"set_keep": "exclude" if a.get("keep") is False else "keep", "set_label": f"label '{a.get('label')}'",
            "set_role": f"role '{a.get('role')}'", "set_audit_kind": f"audit kind '{a.get('audit_kind')}'"}.get(
        p["op"], p.get("op_text") or p["op"])
    return f"{what}" + (f" ({n} column{'s' if n != 1 else ''})" if n else "")


def open_patches(d):
    return [p for p in d.get("patches", []) if p["status"] in ("pending", "held")]


