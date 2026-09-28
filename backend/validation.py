"""Consistency checks (spec 3.6). Run on every AI proposal and on user edits.

Each check compares a claim with the computed data profile. Results:
  ok            nothing to report
  warning       shown to the user; the claim stands
  contradicted  the claim is impossible for this data -> becomes 'unresolved'
"""

from __future__ import annotations

from collections import Counter

from .parsing import cell, is_missing
from .schema import UNRESOLVED, VOCABULARY, kinds_for_role

_LOG_SCALES = {"log2", "log10", "ln"}
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


def column_values(cols, idx):
    return [cell(r, idx).strip() for r in cols.rows]


def timepoint_detail(digest):
    """Deterministic timepoint detail: numeric | date | ordinal_label | None (unclear)."""
    if digest["type"] == "numeric":
        return "numeric"
    if (digest.get("date_like_frac") or 0) >= 0.8:
        return "date"
    if digest.get("n_unique") and digest["n_unique"] <= _ORDINAL_MAX_LEVELS:
        return "ordinal_label"
    return None


def uniqueness(cols, indices):
    """(n_empty, duplicated values with counts) for the key formed by the columns."""
    keys = []
    for r in cols.rows:
        parts = [cell(r, i).strip() for i in indices]
        keys.append(None if any(is_missing(p) for p in parts) else "|".join(parts))
    n_empty = sum(1 for k in keys if k is None)
    counts = Counter(k for k in keys if k is not None)
    dups = [(k, c) for k, c in counts.most_common() if c > 1]
    return n_empty, dups


def validate_group(item, group, cols, layout=None):
    """Validate one group-level claim. Mutates nothing; returns the result."""
    res = _result()
    role = item.get("role") or UNRESOLVED
    if role not in VOCABULARY["column_role"]:
        _contradict(res, f"'{role}' is not an allowed role.")
        return res
    if role == UNRESOLVED:
        return res
    kind = item.get("kind")
    allowed = kinds_for_role(role)
    if allowed and kind not in allowed:
        _contradict(res, f"'{kind}' is not a valid kind for role '{role}'.")
        return res
    prof = group.get("profile") or {}
    numeric = group["type"] == "numeric"

    if role == "value":
        if not numeric:
            _contradict(res, "Contains non-numeric values, so it cannot be a measurement column.")
            return res
        mt, scale, label = item.get("measurement_type"), item.get("scale"), (item.get("label") or "")
        if mt and mt not in VOCABULARY["measurement_type"]:
            _contradict(res, f"'{mt}' is not an allowed measurement type.")
        if scale and scale not in VOCABULARY["scale"]:
            _contradict(res, f"'{scale}' is not an allowed scale.")
        if mt == "count" and not (prof.get("integer_valued") and (prof.get("min") or 0) >= 0):
            _contradict(res, "Proposed as counts, but the values are not all non-negative whole numbers.")
        if mt == "proportion_or_relative_abundance":
            hi = 100 if ("%" in label or "percent" in label.lower()) else 1
            if (prof.get("min") or 0) < 0 or (prof.get("max") or 0) > hi:
                _contradict(res, f"Proposed as proportions, but values fall outside 0-{hi}.")
        if mt == "intensity" and scale in _LOG_SCALES and (prof.get("p99") or 0) > 100:
            _warn(res, f"99th percentile is {prof.get('p99'):g}, unusually large for {scale}-scale values.")
        if mt == "intensity" and scale == "linear" and prof.get("median") is not None \
                and prof["median"] < 100 and (prof.get("log10_span") or 0) < 1.5:
            _warn(res, "Narrow range of small values for linear intensities; the data may already be log-transformed.")
        if prof.get("frac_negative") and scale == "linear" and mt in ("intensity", "count", "concentration"):
            _warn(res, f"{prof['frac_negative']:.0%} of values are negative, unusual for linear {mt}.")
        return res

    if group["n_columns"] != 1 and role in ("feature_id", "sample_id"):
        _warn(res, "An identifier should be a single column; this group has several.")
        return res

    idx = group["indices"][0] if group["n_columns"] == 1 else None
    if idx is not None and role in ("feature_id", "sample_id") and layout != "long":
        n_empty, dups = uniqueness(cols, [idx])
        what = "feature" if role == "feature_id" else "sample"
        if dups:
            shown = ", ".join(f"'{k}' x{c}" for k, c in dups[:5])
            _warn(res, f"{len(dups)} {what} ID value(s) occur more than once ({shown}). Nothing is removed or merged.")
        if n_empty:
            _warn(res, f"{n_empty} row(s) have an empty {what} ID.")

    if role == "sample_metadata" and idx is not None:
        d = cols.digests[idx]
        if kind == "subject_id":
            if d.get("n_unique") is not None and d.get("unique_ratio") == 1:
                _warn(res, "Every value is different; this looks more like a sample ID than a subject ID.")
        if kind == "timepoint":
            detail = item.get("detail")
            auto = timepoint_detail(d)
            if auto is None:
                _warn(res, "Not numeric, not dates, and many different labels: check this is a time point.")
            elif detail and detail != auto:
                _warn(res, f"Values look like '{auto}', not '{detail}'.")
        if kind in ("covariate_numeric", "technical_numeric", "run_order") and d["type"] != "numeric":
            _warn(res, f"'{kind}' expects numbers, but this column is {d['type']}.")
    return res


def validate_feature_identity(fi, groups_by_id, cols, layout=None):
    res = _result()
    gids = fi.get("group_ids") or []
    idxs = []
    for gid in gids:
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
