"""Facts about the table: column digests, shared name parts, layout hints.

Everything here is computed, not judged. Which columns belong together is a
judgment and is proposed by the AI (ai.py) or by you; grouping.py only applies
such proposals. The one number that shapes the AI calls, the chunk size, is a
prompt size limit (config.GROUPING_CHUNK_SIZE), not a claim about families.
"""

from __future__ import annotations

import bisect
import math
import re
from collections import Counter

from .parsing import cell, column_labels, is_missing, parse_number

TEXT_EXAMPLES_MAX_UNIQUE = 20
HIST_BINS = 30
FILE_EXTENSIONS = (".raw", ".mzml", ".mzxml", ".d", ".wiff", ".wiff2", ".dia", ".mgf")
_DATE = re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?$|^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$")


# ---------------------------------------------------------------- helpers

def pct(sorted_vals, q):
    if not sorted_vals:
        return None
    return sorted_vals[int(round(q * (len(sorted_vals) - 1)))]


def _r(x, nd=6):
    if x is None or isinstance(x, bool):
        return x
    if isinstance(x, float):
        if math.isinf(x) or math.isnan(x):
            return None
        return float(f"{x:.{nd}g}")
    return x


def numeric_stats(values, n_total):
    """Stats over parsed floats (missing already removed). n_total = cells incl. missing."""
    vals = sorted(v for v in values if not math.isnan(v))
    n = len(vals)
    if n == 0:
        return {"n": 0, "frac_na": 1.0 if n_total else 0.0}
    finite = [v for v in vals if not math.isinf(v)]
    pos = [v for v in finite if v > 0]
    p1p, p99p = pct(pos, 0.01), pct(pos, 0.99)
    return {
        "n": n,
        "min": _r(vals[0]), "p1": _r(pct(vals, 0.01)), "median": _r(pct(vals, 0.5)),
        "p99": _r(pct(vals, 0.99)), "max": _r(vals[-1]),
        "frac_na": _r((n_total - n) / n_total) if n_total else 0.0,
        "frac_zero": _r(sum(1 for v in vals if v == 0) / n),
        "frac_negative": _r(sum(1 for v in vals if v < 0) / n),
        "integer_valued": all(v == int(v) for v in finite) if finite else False,
        "n_unique": len(set(vals)),
        "log10_span": _r(math.log10(p99p / p1p)) if (p1p and p99p and p1p > 0) else None,
    }


def histogram(values):
    """30-bin histogram of log10(positive values) plus zero / negative counts."""
    pos = [math.log10(v) for v in values if v > 0 and not math.isinf(v)]
    out = {"bins": [], "lo": None, "hi": None, "n_zero": sum(1 for v in values if v == 0),
           "n_negative": sum(1 for v in values if v < 0)}
    if not pos:
        return out
    lo, hi = min(pos), max(pos)
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    counts = [0] * HIST_BINS
    w = (hi - lo) / HIST_BINS
    for x in pos:
        counts[min(int((x - lo) / w), HIST_BINS - 1)] += 1
    out.update(bins=counts, lo=_r(lo), hi=_r(hi))
    return out


# ---------------------------------------------------------------- column digest

class Columns:
    """Parsed per-column data, computed once per table."""

    def __init__(self, table):
        self.header = table["header"]
        self.rows = table["rows"]
        self.labels = column_labels(self.header)
        report = table["parse_report"]
        self.decimal_comma = set(report.get("decimal_comma_columns", []))
        self.n_rows = len(self.rows)
        self._nums = {}
        self.digests = [self._digest(i) for i in range(len(self.header))]

    def raw(self, i):
        return [cell(r, i) for r in self.rows]

    def nums(self, i):
        """Parsed floats for column i (missing / non-numeric cells dropped)."""
        if i not in self._nums:
            dc = self.labels[i] in self.decimal_comma
            out = []
            for v in self.raw(i):
                x = parse_number(v, dc)
                if x is not None:
                    out.append(x)
            self._nums[i] = out
        return self._nums[i]

    def _digest(self, i):
        values = self.raw(i)
        present = [v for v in values if not is_missing(v)]
        n_total, n_present = len(values), len(present)
        d = {"column": self.labels[i], "position": i + 1,
             "frac_na": _r((n_total - n_present) / n_total) if n_total else 0.0}
        nums = self.nums(i)
        if n_present and len(nums) == n_present:
            d["type"] = "numeric"
            d.update({k: v for k, v in numeric_stats(nums, n_total).items() if k not in ("n", "frac_na")})
            return d
        d["type"] = "mixed" if nums else "text"
        counts = Counter(v.strip() for v in present)
        n_unique = len(counts)
        d.update({
            "n_unique": n_unique,
            "unique_ratio": _r(n_unique / n_present) if n_present else None,
            "repeat_rate": _r(sum(c for c in counts.values() if c > 1) / n_present) if n_present else None,
            "min_len": min((len(v) for v in counts), default=0),
            "max_len": max((len(v) for v in counts), default=0),
            "date_like_frac": _r(sum(c for v, c in counts.items() if _DATE.match(v)) / n_present) if n_present else 0,
        })
        if nums:
            d["frac_numeric"] = _r(len(nums) / n_present)
        if 0 < n_unique <= TEXT_EXAMPLES_MAX_UNIQUE:
            d["values"] = [{"value": v, "count": c} for v, c in counts.most_common()]
        d["value_shapes"] = value_shapes(counts)
        return d

    def is_numeric(self, i):
        return self.digests[i]["type"] == "numeric"


# ---------------------------------------------------------------- name patterns

_LEAD = re.compile(r"^[^0-9]{1,6}")


def _mask(s):
    s = re.sub(r"[0-9]", "#", s)
    s = re.sub(r"[a-z]", "a", s)
    return re.sub(r"[A-Z]", "A", s)


def value_shapes(counts, top=3):
    """The format of a text column's values without the values themselves:
    digits -> '#', letters -> 'a' / 'A'. A leading non-digit prefix shared by at
    least half of the values (e.g. 'cg', 'OTU_', 'ENSG') is kept as is, since it
    names the ID system rather than an individual entry."""
    n = sum(counts.values())
    if not n:
        return []
    leads = Counter()
    for v, c in counts.items():
        m = _LEAD.match(v)
        if m and re.search(r"[0-9]", v):
            leads[m.group(0)] += c
    shared = {p for p, c in leads.items() if c >= 0.5 * n}
    shapes = Counter()
    for v, c in counts.items():
        m = _LEAD.match(v)
        if m and m.group(0) in shared:
            shapes[m.group(0) + _mask(v[m.end():])] += c
        else:
            shapes[_mask(v)] += c
    return [{"shape": s[:40], "count": c} for s, c in shapes.most_common(top)]


def strip_rule(names, pattern=None):
    """What to strip from column names to get sample names: the family pattern
    itself, plus a shared directory path and a shared file extension (.raw,
    .mzML, ...). Nothing else is removed, so 'Sample_01' stays 'Sample_01'."""
    pre, suf = "", ""
    if pattern:
        if pattern["side"] == "prefix":
            pre = _cut_to_boundary(pattern["text"], names, "prefix")
        else:
            suf = _cut_to_boundary(pattern["text"], names, "suffix")
    if not pre:
        cp = _common_prefix(names) if len(names) > 1 else ""
        cut = max(cp.rfind("/"), cp.rfind("\\"))
        if cut >= 0:
            pre = cp[:cut + 1]
    if not suf and all(n.lower().endswith(FILE_EXTENSIONS) for n in names):
        ext = "." + names[0].rsplit(".", 1)[-1]
        if all(n.endswith(ext) for n in names):
            suf = ext
    if any(not n[len(pre):len(n) - len(suf)].strip() for n in names):
        return {"strip_prefix": "", "strip_suffix": ""}
    return {"strip_prefix": pre, "strip_suffix": suf}


def _common_prefix(strs):
    s1, s2 = min(strs), max(strs)
    k = 0
    while k < len(s1) and k < len(s2) and s1[k] == s2[k]:
        k += 1
    return s1[:k]


def _cut_to_boundary(p, names, side):
    """Shorten p so the remaining sample name starts/ends at a separator."""
    if not p:
        return p
    def ok(q):
        if not q:
            return True
        edge = q[-1] if side == "prefix" else q[0]
        return not edge.isalnum()
    while p and not ok(p):
        p = p[:-1] if side == "prefix" else p[1:]
    return p


def apply_rule(name, rule):
    """Sample ID from a column name: strip the given prefix / suffix, then add an
    optional per-block label (used to tell apart blocks whose IDs would collide)."""
    pre, suf = rule.get("strip_prefix", ""), rule.get("strip_suffix", "")
    if pre and name.startswith(pre):
        name = name[len(pre):]
    if suf and name.endswith(suf):
        name = name[:len(name) - len(suf)]
    return (rule.get("add_prefix") or "") + name


# ---------------------------------------------------------------- shared name parts (a fact, not a grouping)

def _lcp(a, b):
    k, n = 0, min(len(a), len(b))
    while k < n and a[k] == b[k]:
        k += 1
    return k


def _count_with_prefix(sorted_names, p):
    """How many names start with p (binary search on the sorted list)."""
    return bisect.bisect_left(sorted_names, p + "\U0010ffff") - bisect.bisect_left(sorted_names, p)


def _shared_levels(names):
    """For each name: its longest common prefix with EVERY other name, summarised as
    levels [(text, n_others)] from longest to shortest: how many other names share at
    least that much. Sorted names, the LCP of neighbours, and nearest-smaller links,
    so each name costs O(number of levels). No threshold, no minimum group size."""
    n = len(names)
    order = sorted(range(n), key=lambda i: names[i])
    srt = [names[i] for i in order]
    adj = [_lcp(srt[k], srt[k + 1]) for k in range(n - 1)]   # adj[k]: between sorted k and k+1
    m = len(adj)
    prev_smaller, next_smaller, stack = [-1] * m, [m] * m, []
    for k in range(m):
        while stack and adj[stack[-1]] >= adj[k]:
            stack.pop()
        prev_smaller[k] = stack[-1] if stack else -1
        stack.append(k)
    stack = []
    for k in range(m - 1, -1, -1):
        while stack and adj[stack[-1]] >= adj[k]:
            stack.pop()
        next_smaller[k] = stack[-1] if stack else m
        stack.append(k)
    out = [None] * n
    for pos, i in enumerate(order):
        counts = Counter()
        j = pos - 1                      # names left of pos: LCP = min(adj[q..pos-1])
        while j >= 0 and adj[j] > 0:
            counts[adj[j]] += j - prev_smaller[j]
            j = prev_smaller[j]
        j = pos                          # names right of pos: LCP = min(adj[pos..q-1])
        while j < m and adj[j] > 0:
            counts[adj[j]] += next_smaller[j] - j
            j = next_smaller[j]
        levels, total = [], 0
        for length in sorted(counts, reverse=True):
            total += counts[length]
            levels.append({"text": srt[pos][:length], "n_others": total})
        out[i] = levels
    return out


def shared_affixes(labels):
    """Per column: the prefixes and suffixes it shares with other column names, each
    with how many other columns share it (longest first). This describes the names;
    it does not group anything."""
    pre = _shared_levels(labels)
    suf = _shared_levels([s[::-1] for s in labels])
    return [{"shared_prefix": p, "shared_suffix": [{"text": x["text"][::-1], "n_others": x["n_others"]} for x in s]}
            for p, s in zip(pre, suf)]


def affix_key(a):
    """Lengths of the longest shared prefix / suffix (for ordering columns into chunks)."""
    p = len(a["shared_prefix"][0]["text"]) if a.get("shared_prefix") else 0
    s = len(a["shared_suffix"][0]["text"]) if a.get("shared_suffix") else 0
    return p, s


CALL_COST = 3  # in shared characters: tie-breaker between one more call and a cut through a name part

_DIGITS = re.compile(r"[0-9]+")


def name_template(name):
    """Every run of digits becomes '#', everything else (case included) is kept:
    'Amino Acid_100000011' -> 'Amino Acid_#', 'Pt003_visit1' -> 'Pt#_visit#', 'seq.1234.56' -> 'seq.#.#'."""
    return _DIGITS.sub("#", name)


def ties_at_min(vals):
    """Share of the non-missing values equal to the column's (feature's) minimum."""
    if not vals:
        return None
    m = min(vals)
    return sum(1 for v in vals if v == m) / len(vals)


def name_templates(cols, indices):
    """Name templates of the given columns (a fact, no grouping decision): template ->
    member indices in file order, with an aggregate profile of the members."""
    by = {}
    for i in indices:
        by.setdefault(name_template(cols.labels[i]), []).append(i)
    out = []
    for tpl, members in by.items():
        types = Counter(cols.digests[i]["type"] for i in members)
        x = {"template": tpl, "n_columns": len(members), "indices": members, "types": dict(types)}
        num = [cols.digests[i] for i in members if cols.digests[i]["type"] == "numeric"]
        if num:
            meds = sorted(d["median"] for d in num if d.get("median") is not None)
            p1s = sorted(d["p1"] for d in num if d.get("p1") is not None)
            p99s = sorted(d["p99"] for d in num if d.get("p99") is not None)
            ties = [ties_at_min(cols.nums(i)) for i in members if cols.digests[i]["type"] == "numeric"]
            x["aggregate"] = {
                "median_of_column_medians": _r(pct(meds, 0.5)), "column_medians_range": [_r(meds[0]), _r(meds[-1])] if meds else None,
                "p1": _r(pct(p1s, 0.5)), "p99": _r(pct(p99s, 0.5)),
                "share_integer_valued": _r(sum(1 for d in num if d.get("integer_valued")) / len(num)),
                "share_with_ties_at_minimum": _r(sum(1 for v, i in zip(ties, members) if v is not None
                                                     and v * len(cols.nums(i)) >= 2) / len(num)),
                "frac_zero": _r(sum(d.get("frac_zero") or 0 for d in num) / len(num)),
                "frac_na": _r(sum(d.get("frac_na") or 0 for d in num) / len(num))}
        out.append(x)
    return out


def chunk_units(keys, size):
    """Split units (columns or name templates, given by their sort key) into chunks of at
    most `size` units; boundaries go where neighbouring keys share the least text."""
    n0 = len(keys)
    if n0 <= size:
        return [list(range(n0))]
    order = sorted(range(n0), key=lambda k: keys[k])
    n = len(order)
    cut_cost = [0] + [_lcp(keys[order[k - 1]], keys[order[k]]) for k in range(1, n)]
    best, prev = [0] + [None] * n, [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(max(0, j - size), j):
            if best[i] is None:
                continue
            c = best[i] + CALL_COST + (cut_cost[i] if i else 0)
            if best[j] is None or c < best[j]:
                best[j], prev[j] = c, i
    bounds, j = [], n
    while j > 0:
        bounds.append((prev[j], j))
        j = prev[j]
    return [sorted(order[i:j]) for i, j in reversed(bounds)]


def chunk_columns(indices, labels, affixes, size):
    """Split columns into chunks of at most `size` for separate AI calls.

    `size` is a prompt size limit, not a claim about family size. Columns that share
    a literal prefix / suffix are placed next to each other (sorted by name, or by
    reversed name when the shared suffix is the longer one), and boundaries go
    where neighbouring names share the least text, so families usually stay in one
    chunk. When one does not, the consolidation call
    lets the AI say which groups from different chunks are one family."""
    indices = list(indices)
    if len(indices) <= size:
        return [indices]
    def key(i):
        p, s = affix_key(affixes[i])
        return ("s", labels[i][::-1]) if s > p else ("p", labels[i])
    order = sorted(indices, key=key)
    def overlap(x, y):
        kx, ky = key(x), key(y)
        return _lcp(kx[1], ky[1]) if kx[0] == ky[0] else 0
    # exact DP over cut points: each cut costs the number of characters the two
    # neighbouring names share, each chunk (one more AI call) costs CALL_COST; so an
    # extra call is preferred over cutting through a long shared name part
    n = len(order)
    cut_cost = [0] + [overlap(order[k - 1], order[k]) for k in range(1, n)]
    big = CALL_COST
    best, prev = [0] + [None] * n, [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(max(0, j - size), j):
            if best[i] is None:
                continue
            c = best[i] + big + (cut_cost[i] if i else 0)
            if best[j] is None or c < best[j]:
                best[j], prev[j] = c, i
    bounds, j = [], n
    while j > 0:
        bounds.append((prev[j], j))
        j = prev[j]
    return [sorted(order[i:j]) for i, j in reversed(bounds)]


# ---------------------------------------------------------------- groups (structure only)

def common_pattern(names):
    """The literal text all names share at the start or at the end (whichever is longer):
    a description of a group, used for display and to derive sample names."""
    if len(names) < 2:
        return None
    pre = _common_prefix(names)
    rev = [n[::-1] for n in names]
    suf = _common_prefix(rev)[::-1]
    if not pre.strip() and not suf.strip():
        return None
    return {"side": "prefix", "text": pre} if len(pre) >= len(suf) else {"side": "suffix", "text": suf}


def make_group(cols, group_id, indices, origin, **extra):
    """A group record for the given columns. Grouping decisions are made elsewhere
    (the AI proposal, a signature, or you); this only describes the result."""
    members = sorted(indices)
    types = {cols.digests[i]["type"] for i in members}
    typ = cols.digests[members[0]]["type"] if len(members) == 1 else ("numeric" if types == {"numeric"} else "mixed")
    g = {
        "group_id": group_id,
        "columns": [cols.labels[i] for i in members],
        "indices": members,
        "n_columns": len(members),
        "kind": "single_column" if len(members) == 1 else ("numeric_block" if typ == "numeric" else "column_group"),
        "origin": origin,
        "pattern": common_pattern([cols.header[i] for i in members]),
        "type": typ,
    }
    g.update({k: v for k, v in extra.items() if v is not None})
    if typ == "numeric":
        g["profile"], g["histogram"] = block_profile(cols, members)
    elif len(members) == 1:
        g["profile"] = cols.digests[members[0]]
    else:
        g["profile"] = {"n_columns": len(members), "types": dict(Counter(cols.digests[i]["type"] for i in members))}
    if len(members) > 1 and typ == "numeric":
        names = [cols.header[i] for i in members]
        rule = strip_rule(names, g["pattern"])
        g["sample_id_rule"] = rule
        g["sample_names"] = [apply_rule(nm, rule) for nm in names]
    return g


def block_profile(cols, members):
    vals, n_total = [], 0
    sums = []
    for i in members:
        v = cols.nums(i)
        vals.extend(v)
        n_total += cols.n_rows
        finite = [x for x in v if not math.isinf(x)]
        sums.append(sum(finite))
    st = numeric_stats(vals, n_total)
    st.pop("n", None)
    st["n_columns"] = len(members)
    st["n_rows"] = cols.n_rows
    if len(sums) >= 2:
        m = sum(sums) / len(sums)
        sd = math.sqrt(sum((s - m) ** 2 for s in sums) / (len(sums) - 1))
        st["sample_sum_cv"] = _r(sd / m) if m else None
    else:
        st["sample_sum_cv"] = None
    return st, histogram(vals)


# ---------------------------------------------------------------- layout hints

def layout_hints(cols):
    """Facts about the table's shape; the AI (or you) decides the layout."""
    n_rows, n_cols = cols.n_rows, len(cols.header)
    numeric = [d for d in cols.digests if d["type"] == "numeric"]
    text_cols = [d for d in cols.digests if d["type"] != "numeric"]
    repeated_text = [d for d in text_cols if (d.get("repeat_rate") or 0) > 0.5 and (d.get("n_unique") or 0) > 1]
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "numeric_columns": len(numeric),
        "rows_to_numeric_columns_ratio": round(n_rows / len(numeric), 3) if numeric else None,
        "repeated_text_columns": len(repeated_text),
        "long_format_pattern": bool(len(repeated_text) >= 2 and 1 <= len(numeric) <= 3),
        "note": ("Features usually outnumber samples: many rows and few numeric columns suggests samples in "
                 "columns; few rows and very many numeric columns suggests samples in rows."),
    }


def profile_table(table):
    cols = Columns(table)
    return cols, shared_affixes(cols.labels), layout_hints(cols)
