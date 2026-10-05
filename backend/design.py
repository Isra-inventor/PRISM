"""Study design facts (v2.4 §7). Computed by code, never judged by the AI.

Where subject and time come from is proposed (by the AI or you) and confirmed by you:
    source: metadata_column | derived_from_sample_names | none
A derivation is a small rule over the sample names (derive.py), previewed with its
coverage. Once sources are chosen, code computes: samples per subject, subjects with a
single sample, balanced yes / no, repeated_measures.detected (any subject with >= 2
samples: later steps branch on it, so it is computed, not judged), the time kind
(numeric / ordinal_label / date) and distinct values, per-subject series, which sample
columns vary within a subject, and cross-checks between several possible sources.
The time unit is always asked (a question), never inferred.
"""

from __future__ import annotations

import re
from collections import Counter, OrderedDict

from . import derive
from .parsing import cell, is_missing, parse_number

SOURCES = ("metadata_column", "derived_from_sample_names", "none")
PARTS = ("subject", "time", None, "")
UNITS = ["hours", "days", "weeks", "months", "other", "not time-based, just an index"]
_DATE = re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$")
_DELIMS = ["_", "-", ".", " ", "|", ":"]


def blank():
    return {"subject": {"source": "none", "column": None, "file": None},
            "time": {"source": "none", "column": None, "file": None, "unit": {"value": None, "provenance": "unanswered"}},
            "derivation": None}


def derivation_candidates(names, top=6):
    """For each literal delimiter and first / last occurrence: how many names it splits and how
    many distinct values each side has. A fact for the AI and the UI, not a choice."""
    out = []
    for dl in _DELIMS:
        for occ in ("first", "last"):
            res = [derive.split_name(n, dl, occ) for n in names]
            ok = [r for r in res if r]
            if not ok:
                continue
            out.append({"delimiter": dl, "occurrence": occ, "n_parsed": len(ok), "n_total": len(names),
                        "n_distinct_left": len({r[0] for r in ok}), "n_distinct_right": len({r[1] for r in ok}),
                        "right_all_numeric": all(re.fullmatch(r"\d+(\.\d+)?", r[1]) for r in ok)})
    out.sort(key=lambda x: (-x["n_parsed"], x["delimiter"] != "_", x["occurrence"] != "last"))
    seen, uniq = set(), []
    for x in out:   # first and last coincide when the delimiter occurs once
        k = (x["delimiter"], x["n_parsed"], x["n_distinct_left"], x["n_distinct_right"])
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    return uniq[:top]


def check_derivation(rule):
    derive.check_rule(rule)
    sides = (rule.get("left"), rule.get("right"))
    if any(x not in PARTS for x in sides):
        raise derive.RuleError("left and right must each be 'subject', 'time' or empty.")
    if not any(sides):
        raise derive.RuleError("Say which side is the subject and / or the time.")
    if sides[0] and sides[0] == sides[1]:
        raise derive.RuleError("left and right cannot both be the same thing.")


def derived_values(ids, rule):
    """{sample: {"subject": ..., "time": ...}} for the names the rule parses, and the failures."""
    out, fails = {}, []
    for sid in ids:
        parts = derive.split_name(sid, rule["delimiter"], rule["occurrence"])
        if parts is None:
            fails.append(sid)
            continue
        v = {}
        for side, val in zip((rule.get("left"), rule.get("right")), parts):
            if side:
                v[side] = val
        out[sid] = v
    return out, fails


def preview(ids, rule, limit=200):
    check_derivation(rule)
    vals, fails = derived_values(ids, rule)
    rows = [{"sample": sid, "subject": vals.get(sid, {}).get("subject"), "time": vals.get(sid, {}).get("time")}
            for sid in ids]
    return {"rule": {k: rule.get(k) for k in ("delimiter", "occurrence", "left", "right")},
            "n_total": len(ids), "n_parsed": len(vals), "coverage": round(len(vals) / len(ids), 4) if ids else 0.0,
            "failures": fails[:30], "n_failures": len(fails), "rows": rows[:limit]}


def time_kind(values):
    vals = [v for v in values if v not in (None, "")]
    if not vals:
        return None
    if all(parse_number(v) is not None for v in vals):
        return "numeric"
    if sum(1 for v in vals if _DATE.match(v)) >= 0.8 * len(vals):
        return "date"
    return "ordinal_label"


def _sort_time(v):
    x = parse_number(v) if v is not None else None
    return (0, x, "") if x is not None else (1, 0, str(v))


def summarize(per_sample):
    """per_sample: {sample: {"subject": s, "time": t}} -> computed design facts."""
    subj = OrderedDict()
    for sid, v in per_sample.items():
        s = v.get("subject")
        if s not in (None, ""):
            subj.setdefault(s, []).append(sid)
    counts = [len(v) for v in subj.values()]
    out = {}
    if subj:
        dist = Counter(counts)
        out["repeated_measures"] = {
            "detected": any(c >= 2 for c in counts), "provenance": "computed",
            "samples_per_subject": {"min": min(counts), "max": max(counts),
                                    "counts": {str(k): dist[k] for k in sorted(dist)}},
            "subjects_with_single_sample": sum(1 for c in counts if c == 1),
            "balanced": len(set(counts)) == 1,
            "per_subject": {s: len(v) for s, v in subj.items()}}
        out["n_subjects"] = len(subj)
        out["n_samples_with_subject"] = sum(counts)
    times = [v.get("time") for v in per_sample.values() if v.get("time") not in (None, "")]
    if times:
        out["time_kind"] = time_kind(times)
        out["n_distinct_time"] = len(set(times))
        out["time_values"] = sorted(set(times), key=_sort_time)[:60]
    if subj and times:
        out["series"] = {s: sorted(([sid, per_sample[sid].get("time")] for sid in sids), key=lambda x: _sort_time(x[1]))
                         for s, sids in list(subj.items())[:200]}
    out["label"] = design_label(out)
    return out


def design_label(f):
    rm = f.get("repeated_measures")
    if not rm:
        return "no subject information"
    sp = rm["samples_per_subject"]
    if not rm["detected"]:
        return "cross-sectional (one sample per subject)"
    kind = "longitudinal" if f.get("n_distinct_time") else "repeated measures"
    if rm["balanced"]:
        return f"{kind}, balanced ({sp['min']} samples per subject)"
    return f"{kind}, unbalanced ({sp['min']} to {sp['max']} samples per subject)"


def varies_within_subject(per_sample, column_values):
    """column_values: {sample: value}. true / false over subjects with >= 2 samples, else not_applicable."""
    subj = {}
    for sid, v in per_sample.items():
        if v.get("subject") not in (None, ""):
            subj.setdefault(v["subject"], []).append(sid)
    multi = [sids for sids in subj.values() if len(sids) >= 2]
    if not multi:
        return "not_applicable"
    for sids in multi:
        vals = {column_values.get(x) for x in sids if column_values.get(x) not in (None, "")}
        if len(vals) > 1:
            return True
    return False


def cross_check(name, a, b):
    """Two sources of the same thing (e.g. derived subject vs a metadata subject column), over the
    samples where both have a value: agree when they partition the samples the same way (the
    values may be written differently: DEAB vs NHP-DEAB, 0 vs W0)."""
    both = [s for s in a if a[s] not in (None, "") and b.get(s) not in (None, "")]
    if not both:
        return {"check": name, "agree": 0, "disagree": 0, "n_compared": 0}
    fwd, back = {}, {}
    for s in both:
        fwd.setdefault(a[s], Counter())[b[s]] += 1
        back.setdefault(b[s], Counter())[a[s]] += 1
    best = {k: c.most_common(1)[0][0] for k, c in fwd.items()}
    bestb = {k: c.most_common(1)[0][0] for k, c in back.items()}
    agree = sum(1 for s in both if best[a[s]] == b[s] and bestb[b[s]] == a[s])
    out = {"check": name, "agree": agree, "disagree": len(both) - agree, "n_compared": len(both)}
    bad = [s for s in both if not (best[a[s]] == b[s] and bestb[b[s]] == a[s])]
    if bad:
        out["examples"] = [{"sample": s, "a": a[s], "b": b[s]} for s in bad[:5]]
    return out


def column_values_main(s, sample_ids, gid):
    """{sample: value} of a main-file sample column (samples in rows: one row per sample)."""
    i = s.groups_by_id[gid]["indices"][0]
    return {sid: ("" if is_missing(cell(r, i)) else cell(r, i).strip()) for sid, r in zip(sample_ids, s.table["rows"])}


def column_values_meta(s, d, column):
    """{sample: value} of a metadata-file column, joined on the confirmed key (and accepted near misses)."""
    meta = d.get("metadata") or {}
    if not s.metadata_table or meta.get("skipped"):
        return {}
    t = s.metadata_table["table"]
    by = {c["column"]: c for c in meta["columns"]}
    if column not in by or meta.get("id_column") not in by:
        return {}
    ki, ci = by[meta["id_column"]]["index"], by[column]["index"]
    alias = {m: dd for dd, m in meta.get("accepted_near_misses", [])}
    out = {}
    for r in t["rows"]:
        mid = cell(r, ki).strip()
        v = cell(r, ci)
        out.setdefault(alias.get(mid, mid), "" if is_missing(v) else v.strip())
    return out
