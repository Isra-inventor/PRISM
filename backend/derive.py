"""The small rule language for splitting names (v2.4 §7.2 and §17). No free regex.

A rule splits a name at the first or the last occurrence of a literal delimiter
into a left and a right part:
    sample names:  {"delimiter": "_", "occurrence": "last", "left": "subject", "right": "time"}
    feature names: {"delimiter": "_", "occurrence": "first",
                    "parts": [{"name": "feature_class", ...}, {"name": "feature_numeric_id", ...}]}
A name without the delimiter is a failure (reported, never guessed). Every
derivation is previewed with its coverage before anything is recorded.
"""

from __future__ import annotations

import re
from collections import Counter

OCCURRENCES = ("first", "last")
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,39}$")


class RuleError(ValueError):
    pass


def check_rule(rule):
    if not isinstance(rule, dict):
        raise RuleError("A rule needs a delimiter and an occurrence (first or last).")
    delim = rule.get("delimiter")
    if not isinstance(delim, str) or not delim or len(delim) > 5:
        raise RuleError("The delimiter must be 1 to 5 literal characters (no regex).")
    if rule.get("occurrence") not in OCCURRENCES:
        raise RuleError("occurrence must be 'first' or 'last'.")
    return delim, rule["occurrence"]


def split_name(name, delimiter, occurrence):
    """(left, right) or None when the delimiter does not occur."""
    if delimiter not in name:
        return None
    left, _, right = name.partition(delimiter) if occurrence == "first" else name.rpartition(delimiter)
    return left, right


def check_parts(rule, taken=()):
    parts = rule.get("parts") or []
    if len(parts) != 2:
        raise RuleError("Name the two parts (left and right of the delimiter); leave a name empty to drop a part.")
    names = [(p or {}).get("name") or "" for p in parts]
    if not any(names):
        raise RuleError("Name at least one part.")
    for n in names:
        if n and not _NAME.match(n):
            raise RuleError(f"Part name '{n}' must be lower case letters, digits and underscores.")
        if n and n in taken:
            raise RuleError(f"'{n}' is already a column name.")
    if names[0] and names[0] == names[1]:
        raise RuleError("The two parts need different names.")
    return names


def apply(names, rule):
    delim, occ = check_rule(rule)
    return [split_name(n, delim, occ) for n in names]


def preview(names, rule, part_names=("left", "right"), top=30):
    """Coverage, failures and the distinct values of each part (with counts)."""
    res = apply(names, rule)
    ok = [r for r in res if r is not None]
    fails = [n for n, r in zip(names, res) if r is None]
    parts = {}
    for k, pn in enumerate(part_names):
        if not pn:
            continue
        c = Counter(r[k] for r in ok)
        parts[pn] = {"n_distinct": len(c), "n_empty": c.get("", 0),
                     "values": [{"value": v, "label": "(empty)" if v == "" else v, "count": n}
                                for v, n in c.most_common(top)]}
    return {"n_total": len(names), "n_parsed": len(ok), "coverage": round(len(ok) / len(names), 4) if names else 0.0,
            "failures": fails[:20], "n_failures": len(fails), "parts": parts,
            "examples": [{"name": n, "left": r[0], "right": r[1]} if r else {"name": n, "left": None, "right": None}
                         for n, r in list(zip(names, res))[:8]]}
