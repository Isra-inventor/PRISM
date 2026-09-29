"""Structural checks (v2.1 §6). Run on every AI proposal and on user edits.

Kept on purpose: closed-field values must be in their closed sets, role
'value' requires numeric cells, and ID columns are checked for uniqueness
(warning only, nothing is ever deduplicated). Descriptive labels are free
text and are not checked; the UI shows the computed profile next to them.

Results: ok | warning (shown, claim stands) | contradicted (-> unresolved).
"""

from __future__ import annotations

from collections import Counter

from .parsing import cell, is_missing
from .schema import UNRESOLVED, VOCABULARY

_ORDINAL_MAX_LEVELS = 12


def _result():
    return {"status": "ok", "messages": []}


def _warn(res, msg):
    res["messages"].append(msg)
    if res["status"] == "ok":
        res["status"] = "warning"


def _contradict(res, msg):
    res["messages"].append(msg)
    res["status"] = "contradicted"


def timepoint_detail(digest):
    """A computed default for a time point's free-text detail."""
    if digest["type"] == "numeric":
        return "numeric"
    if (digest.get("date_like_frac") or 0) >= 0.8:
        return "date"
    if digest.get("n_unique") and digest["n_unique"] <= _ORDINAL_MAX_LEVELS:
        return "ordinal label (" + ", ".join(v["value"] for v in (digest.get("values") or [])[:6]) + ")"
    return None


def uniqueness(cols, indices):
    """(n_empty, duplicated key values with counts) for the key formed by the columns."""
    keys = []
    for r in cols.rows:
        parts = [cell(r, i).strip() for i in indices]
        keys.append(None if any(is_missing(p) for p in parts) else "|".join(parts))
    counts = Counter(k for k in keys if k is not None)
    return sum(1 for k in keys if k is None), [(k, c) for k, c in counts.most_common() if c > 1]


def validate_group(item, group, cols, layout=None):
    res = _result()
    role = item.get("role") or UNRESOLVED
    if role not in VOCABULARY["column_role"]:
        _contradict(res, f"'{role}' is not an allowed role.")
        return res
    if role == UNRESOLVED:
        return res
    if role == "value":
        if group["type"] != "numeric":
            _contradict(res, "Contains non-numeric values, so it cannot be a measurement column.")
        return res
    if role == "sample_metadata" and item.get("audit_kind") not in (None, *VOCABULARY["audit_kind"]):
        _contradict(res, f"'{item.get('audit_kind')}' is not an allowed audit kind.")
        return res
    if role in ("feature_id", "sample_id") and group["n_columns"] != 1:
        _warn(res, "An identifier should be a single column; this group has several.")
        return res
    if role in ("feature_id", "sample_id") and layout != "long":
        n_empty, dups = uniqueness(cols, group["indices"])
        what = "feature" if role == "feature_id" else "sample"
        if dups:
            shown = ", ".join(f"'{k}' x{c}" for k, c in dups[:5])
            _warn(res, f"{len(dups)} {what} ID value(s) occur more than once ({shown}). Nothing is removed or merged.")
        if n_empty:
            _warn(res, f"{n_empty} row(s) have an empty {what} ID.")
    return res


def validate_feature_identity(fi, groups_by_id, cols, layout=None):
    res = _result()
    idxs = []
    for gid in fi.get("group_ids") or []:
        g = groups_by_id.get(gid)
        if g is None:
            _contradict(res, f"Unknown group '{gid}'.")
            return res
        idxs.extend(g["indices"])
    if not idxs or layout == "long":  # keys repeat once per sample in a long table
        return res
    n_empty, dups = uniqueness(cols, idxs)
    if dups:
        shown = ", ".join(f"'{k}' x{c}" for k, c in dups[:5])
        _warn(res, f"{len(dups)} feature key(s) occur more than once ({shown}). Nothing is removed or merged.")
    if n_empty:
        _warn(res, f"{n_empty} row(s) have an empty feature key.")
    return res
