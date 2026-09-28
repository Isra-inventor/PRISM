"""Column digests (spec 3.2), column grouping (3.3) and layout hints (3.4).

All of this is deterministic. The AI later labels the groups built here; it
can never create, merge or split them itself.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .parsing import cell, column_labels, is_missing, parse_number

TEXT_EXAMPLES_MAX_UNIQUE = 20
MIN_GROUP_SIZE = 3
MIN_PATTERN_CHARS = 3
MIN_PROFILE_CLUSTER = 5
HIST_BINS = 30
FILE_EXTENSIONS = (".raw", ".mzml", ".mzxml", ".d", ".wiff", ".wiff2", ".dia", ".mgf")
_TOKEN = re.compile(r"[A-Za-z0-9]+")
_SUBTOKEN = re.compile(r"[A-Za-z]+|[0-9]+")
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
        return d

    def is_numeric(self, i):
        return self.digests[i]["type"] == "numeric"


# ---------------------------------------------------------------- name patterns

def _split_points(name, regex):
    return [(m.start(), m.end()) for m in regex.finditer(name)]


def _candidates(name):
    """Prefix / suffix candidates at token boundaries (non-alphanumeric separators
    and letter/digit transitions). Returns [(side, pattern)]."""
    out = set()
    spans = _split_points(name, _SUBTOKEN)
    for k in range(1, len(spans)):
        pre = name[:spans[k][0]]
        if len(pre.strip()) >= MIN_PATTERN_CHARS:
            out.add(("prefix", pre))
        suf = name[spans[k - 1][1]:]
        if len(suf.strip()) >= MIN_PATTERN_CHARS:
            out.add(("suffix", suf))
    return out


def _residual(name, side, pattern):
    r = name[len(pattern):] if side == "prefix" else name[:len(name) - len(pattern)]
    return r.strip(" _.-:|/\\")


def _ntok(s):
    return len(_TOKEN.findall(s))


_SAMPLE_WORDS = re.compile(r"(?i)(^|[^a-z])(qc|pool|pooled|blank|buffer|calib\w*|std|standard)([^a-z]|$)")


def _best_family(names_by_idx, compatible=None):
    """Pick the best name family among the given columns, or None.

    Largest family first (ties: longer pattern). A candidate is skipped when it
    is really a per-sample slice across families (its variable parts are other
    families' patterns) or when it swallows a more specific family whose sample
    names re-appear in the rest (e.g. ' Intensity' vs ' MaxLFQ Intensity')."""
    cands = defaultdict(set)
    for i, name in names_by_idx.items():
        for c in _candidates(name):
            cands[c].add(i)
    cands = {c: m for c, m in cands.items() if len(m) >= MIN_GROUP_SIZE}
    pattern_words = {pat.strip(" _.-:|/\\").lower() for (_, pat) in cands}
    def at_separator(side, pat):
        edge = pat[-1] if side == "prefix" else pat[0]
        return not edge.isalnum()
    ordered = sorted(cands.items(), key=lambda kv: (len(kv[1]), at_separator(*kv[0]), len(kv[0][1].strip())),
                     reverse=True)
    for (side, pat), members in ordered:
        res = [_residual(names_by_idx[i], side, pat) for i in members]
        if any(not r for r in res):
            continue
        if re.search(r"\d", pat) and sum(1 for r in res if r.lower() in pattern_words) >= 0.5 * len(res):
            continue  # a per-sample slice, e.g. suffix ' S01' over 'Intensity S01', 'iBAQ S01', ...
        if _mixes_subfamilies(members, names_by_idx, cands):
            continue
        if compatible is not None and not compatible(members):
            continue
        return side, pat, members
    return None


def _mixes_subfamilies(members, names_by_idx, cands):
    """True if a smaller family inside `members` has (letter-containing) sample
    names that all re-appear among the other members: the candidate mixes
    several measurement families of the same samples."""
    for (s2, p2), m2 in cands.items():
        if not m2 < members:
            continue
        sample_names = {_residual(names_by_idx[i], s2, p2) for i in m2}
        if not all(re.search(r"[A-Za-z]", n) for n in sample_names):
            continue
        others = [" ".join(_TOKEN.findall(names_by_idx[i])) for i in members - m2]
        hits = 0
        for sn in sample_names:
            key = " ".join(_TOKEN.findall(sn))
            if key and any((" " + key + " ") in (" " + o + " ") for o in others):
                hits += 1
        if hits == len(sample_names):
            return True
    return False


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
    pre, suf = rule.get("strip_prefix", ""), rule.get("strip_suffix", "")
    if pre and name.startswith(pre):
        name = name[len(pre):]
    if suf and name.endswith(suf):
        name = name[:len(name) - len(suf)]
    return name


# ---------------------------------------------------------------- grouping

def _col_profile_keys(cols, i):
    d = cols.digests[i]
    med = d.get("median")
    pos = [v for v in cols.nums(i) if v > 0]
    lmed = math.log10(sorted(pos)[len(pos) // 2]) if pos else None
    return {"lmed": lmed, "int": bool(d.get("integer_valued")), "na": d.get("frac_na", 0.0), "median": med}


def _split_deviants(cols, members, exempt_sample_words=False):
    """Split out columns whose profile deviates strongly from their name family.
    In name families, columns named like QC / blank / pool samples are kept:
    those are samples that are expected to look different."""
    if len(members) < 5:
        return members, []
    prof = {i: _col_profile_keys(cols, i) for i in members}
    lm = sorted(p["lmed"] for p in prof.values() if p["lmed"] is not None)
    med = lm[len(lm) // 2] if lm else None
    mad = sorted(abs(x - med) for x in lm)[len(lm) // 2] if lm else 0
    scale = max(1.4826 * mad, 0.1)
    n_int = sum(1 for p in prof.values() if p["int"])
    majority_int = n_int >= 0.8 * len(members)
    majority_float = n_int <= 0.2 * len(members)
    nas = sorted(p["na"] for p in prof.values())
    med_na = nas[len(nas) // 2]
    keep, out = [], []
    for i in members:
        p = prof[i]
        reasons = []
        if exempt_sample_words and _SAMPLE_WORDS.search(cols.header[i]):
            keep.append(i)
            continue
        if med is not None and p["lmed"] is not None and abs(p["lmed"] - med) / scale > 3:
            reasons.append("typical value far from the rest of the family")
        if (majority_int and not p["int"]) or (majority_float and p["int"]):
            reasons.append("integer / non-integer differs from the rest of the family")
        if abs(p["na"] - med_na) > 0.5:
            reasons.append("share of missing values differs strongly")
        (out if reasons else keep).append((i, reasons) if reasons else i)
    if len(keep) < MIN_GROUP_SIZE:
        return members, []
    return keep, out


def _profile_clusters(cols, idxs):
    """Cluster name-less numeric columns by typical value (gap > 1 decade) and
    integer flag. Only clusters of >= MIN_PROFILE_CLUSTER columns are kept."""
    clusters = []
    for is_int in (False, True):
        sub = [(k["lmed"], i) for i in idxs for k in [_col_profile_keys(cols, i)]
               if k["int"] == is_int and k["lmed"] is not None]
        sub.sort()
        cur = []
        for lmed, i in sub:
            if cur and lmed - cur[-1][0] > 1.0:
                clusters.append([j for _, j in cur])
                cur = []
            cur.append((lmed, i))
        if cur:
            clusters.append([j for _, j in cur])
    return [c for c in clusters if len(c) >= MIN_PROFILE_CLUSTER]


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


def build_groups(cols):
    """Return groups[] in file order: numeric name families (with deviants split
    out), profile clusters of the remaining numeric columns, and singletons."""
    n = len(cols.header)
    numeric = [i for i in range(n) if cols.is_numeric(i)]
    remaining = {i: cols.header[i] for i in numeric}
    families = []
    def compatible(members):
        """Small name families (< 5 columns) must also look alike: same integer
        flag and typical values within 1.5 decades (so 'row ID', 'row m/z',
        'row retention time' are not taken for one measurement family)."""
        if len(members) >= 5:
            return True
        prof = [_col_profile_keys(cols, i) for i in members]
        if len({p["int"] for p in prof}) > 1:
            return False
        lm = [p["lmed"] for p in prof if p["lmed"] is not None]
        return not lm or max(lm) - min(lm) <= 1.5

    while True:
        best = _best_family(remaining, compatible)
        if best is None:
            break
        side, pat, members = best
        for i in members:
            remaining.pop(i)
        keep, deviants = _split_deviants(cols, sorted(members), exempt_sample_words=True)
        families.append({"members": keep, "pattern": {"side": side, "text": pat}, "origin": "name_pattern"})
        for i, reasons in deviants:
            families.append({"members": [i], "pattern": None, "origin": "split_from_family",
                             "split_from": pat, "split_reasons": reasons})
    for cl in _profile_clusters(cols, sorted(remaining)):
        for i in cl:
            remaining.pop(i)
        keep, deviants = _split_deviants(cols, sorted(cl))
        families.append({"members": keep, "pattern": None, "origin": "profile_cluster"})
        for i, reasons in deviants:
            families.append({"members": [i], "pattern": None, "origin": "split_from_family",
                             "split_from": "similar-profile cluster", "split_reasons": reasons})
    for i in sorted(remaining):
        families.append({"members": [i], "pattern": None, "origin": "single_numeric"})
    in_family = {i for f in families for i in f["members"]}
    for i in range(n):
        if i not in in_family:
            families.append({"members": [i], "pattern": None, "origin": "single_text"})

    families.sort(key=lambda f: min(f["members"]))
    groups = []
    for k, f in enumerate(families, 1):
        members = sorted(f["members"])
        g = {
            "group_id": f"g{k}",
            "columns": [cols.labels[i] for i in members],
            "indices": members,
            "n_columns": len(members),
            "kind": "numeric_block" if len(members) > 1 else "single_column",
            "origin": f["origin"],
            "pattern": f["pattern"],
            "type": cols.digests[members[0]]["type"] if len(members) == 1 else "numeric",
        }
        if f.get("split_from"):
            g["split_from"] = f["split_from"]
            g["split_reasons"] = f["split_reasons"]
        if g["type"] == "numeric":
            prof, hist = block_profile(cols, members)
            g["profile"], g["histogram"] = prof, hist
        else:
            g["profile"] = cols.digests[members[0]]
        if len(members) > 1 and f["origin"] == "name_pattern":
            names = [cols.header[i] for i in members]
            rule = strip_rule(names, f["pattern"])
            g["sample_id_rule"] = rule
            g["sample_names"] = [apply_rule(nm, rule) for nm in names]
        groups.append(g)
    return groups


# ---------------------------------------------------------------- layout hints

def layout_hints(cols, groups):
    blocks = [g for g in groups if g["kind"] == "numeric_block"]
    largest = max(blocks, key=lambda g: g["n_columns"], default=None)
    n_rows, n_cols = cols.n_rows, len(cols.header)
    text_cols = [d for d in cols.digests if d["type"] != "numeric"]
    repeated_text = [d for d in text_cols if (d.get("repeat_rate") or 0) > 0.5 and (d.get("n_unique") or 0) > 1]
    single_numeric = [d for d in cols.digests if d["type"] == "numeric"]
    hints = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "largest_numeric_block_columns": largest["n_columns"] if largest else 0,
        "largest_numeric_block_share_of_columns": round(largest["n_columns"] / n_cols, 3) if largest else 0,
        "numeric_block_columns_total": sum(g["n_columns"] for g in blocks),
        "rows_to_block_columns_ratio": round(n_rows / sum(g["n_columns"] for g in blocks), 3) if blocks else None,
        "long_format_pattern": bool(len(repeated_text) >= 2 and 1 <= len(single_numeric) <= 3 and not largest),
        "note": ("Features usually outnumber samples: many rows and a numeric block of few columns "
                 "suggests samples in columns; few rows and a very wide numeric block suggests samples in rows."),
    }
    return hints


def profile_table(table):
    cols = Columns(table)
    groups = build_groups(cols)
    return cols, groups, layout_hints(cols, groups)
