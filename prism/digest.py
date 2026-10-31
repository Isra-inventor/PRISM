"""Step 3b: the digest, which is everything the AI is allowed to see about a table.

The AI never gets data rows. It gets, per column, the name, its position and facts computed by
code (kind, missing cells, distinct values, a few statistics), plus for some text columns their
most common values with counts. digest(table) builds it; print it to see exactly what would be sent.

Three rules keep it small and private:

1. Name templates. Columns whose names differ only in their digits ("Amino Acid_1", "Amino Acid_2",
   ... or "S01", "S02", ...) are one entry, with the digits shown as '#', the number of columns,
   and statistics over all of them. A file with 1,000 metabolite columns becomes a few entries.
   Only all-numeric columns are merged this way; text columns are always listed one by one.
2. Example values only from repetitive text columns. A text column where most values are distinct
   (patient codes, protein names, sequence IDs) never sends values, only their "shape"
   (letters as A, digits as 9: "SL000001" -> "AA999999"). A column with few distinct values
   (a group, a flag, an organism) sends its values and counts, so the AI can tell what it is.
   examples=False switches example values off completely.
3. The file name is not sent, only its extension.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter

from .layout import Layout, _numbers
from .table import MISSING, Table

EXAMPLE_MAX_UNIQUE_RATIO = 0.5  # text columns with more distinct values than this share send no values
TOP_VALUES = 12                 # at most this many example values per column
TOP_SHAPES = 3
# Starts of protein IDs that usually mark decoys and contaminants. Only counted (a fact), so the AI
# can propose a flag even for ID columns whose values are never sent.
KNOWN_PREFIXES = ("REV__", "rev_", "CON__", "contam_", "DECOY_", "decoy_", "Cont_")


def name_template(name: str) -> str:
    """The column name with every run of digits replaced by '#'."""
    return re.sub(r"\d+", "#", name)


def _r(x, digits=4):
    """Round to a few significant digits so the digest stays short."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return float(f"{x:.{digits}g}")


def _shape(value: str) -> str:
    return re.sub(r"[0-9]", "9", re.sub(r"[A-Za-z]", "A", value))[:40]


def numeric_facts(values) -> dict:
    """Statistics of one numeric column (values: numbers, NaN for missing)."""
    v = values.dropna()
    if v.empty:
        return {}
    pos = v[v > 0]
    return {
        "min": _r(v.min()), "median": _r(v.median()), "max": _r(v.max()),
        "frac_zero": _r((v == 0).mean(), 3), "frac_negative": _r((v < 0).mean(), 3),
        "integer_valued": bool((v == v.round()).all()),
        # how many powers of ten the positive values span: raw intensities span many, log values few
        "log10_span": _r(math.log10(pos.max() / pos.min()), 3) if len(pos) >= 2 else None,
    }


def column_facts(table: Table, name: str, examples: bool = True) -> dict:
    """What the AI sees about one column."""
    col = next(c for c in table.columns if c.name == name)
    n_rows = table.shape[0]
    d = {"column": name, "position": list(table.data.columns).index(name) + 1, "kind": col.kind,
         "frac_missing": _r(col.n_missing / n_rows, 3) if n_rows else 0.0, "n_unique": col.n_unique}
    if col.kind == "numeric":
        d.update(numeric_facts(_numbers(table.data[name])))
    elif col.kind == "text":
        cells = table.data[name].str.strip()
        counts = Counter(cells[~cells.isin(MISSING)])
        n_present = sum(counts.values())
        d["unique_ratio"] = _r(col.n_unique / n_present, 3)
        d["length"] = [min(map(len, counts)), max(map(len, counts))]
        shapes = Counter()
        for value, count in counts.items():
            shapes[_shape(value)] += count
        d["value_shapes"] = [{"shape": s, "count": c} for s, c in shapes.most_common(TOP_SHAPES)]
        starts = {p: sum(c for v, c in counts.items() if v.startswith(p)) for p in KNOWN_PREFIXES}
        if any(starts.values()):
            d["rows_starting_with"] = {p: n for p, n in starts.items() if n}
        if examples and d["unique_ratio"] <= EXAMPLE_MAX_UNIQUE_RATIO:
            d["values"] = [{"value": v, "count": c} for v, c in counts.most_common(TOP_VALUES)]
            if len(counts) > TOP_VALUES:
                d["values_not_shown"] = len(counts) - TOP_VALUES
    return d


def _template_entry(table: Table, template: str, names: list[str]) -> dict:
    """One entry for several numeric columns that share a name template."""
    facts = [numeric_facts(_numbers(table.data[n])) for n in names]
    facts = [f for f in facts if f]
    medians = sorted(f["median"] for f in facts)
    positions = [list(table.data.columns).index(n) + 1 for n in names]
    shown = names if len(names) <= 8 else names[:3] + [f"... {len(names) - 6} more ..."] + names[-3:]
    entry = {"template": template, "n_columns": len(names), "positions": [positions[0], positions[-1]],
             "columns": shown, "kind": "numeric"}
    if facts:
        entry["over_all_columns"] = {
            "median_of_medians": _r(medians[len(medians) // 2]),
            "range_of_medians": [medians[0], medians[-1]],
            "min": _r(min(f["min"] for f in facts)), "max": _r(max(f["max"] for f in facts)),
            "share_integer_valued": _r(sum(f["integer_valued"] for f in facts) / len(facts), 3),
            "mean_frac_zero": _r(sum(f["frac_zero"] for f in facts) / len(facts), 3),
        }
    missing = [next(c for c in table.columns if c.name == n).n_missing for n in names]
    entry["mean_frac_missing"] = _r(sum(missing) / len(names) / table.shape[0], 3) if table.shape[0] else 0.0
    return entry


def numeric_templates(table: Table) -> dict[str, list[str]]:
    """Name template -> its columns, for templates shared by two or more numeric columns."""
    by_template: dict[str, list[str]] = {}
    for c in table.columns:
        if c.kind == "numeric":
            by_template.setdefault(name_template(c.name), []).append(c.name)
    return {t: names for t, names in by_template.items() if len(names) >= 2}


def digest(table: Table, layout: Layout | None = None, examples: bool = True) -> dict:
    """Everything the AI will see about this table. layout: the step-2 proposal, sent as a hint."""
    merged = numeric_templates(table)
    in_template = {n for names in merged.values() for n in names}

    d = {
        "file": {"extension": table.path.suffix.lower(), "n_rows": table.shape[0], "n_columns": table.shape[1]},
        "templates": [_template_entry(table, t, names) for t, names in merged.items()],
        "columns": [column_facts(table, n, examples) for n in table.data.columns if n not in in_template],
        "settings": {"example_values_sent": examples, "data_rows_sent": False},
    }
    if layout is not None:
        d["file"]["layout_hint"] = {"orientation": layout.orientation, "row_id_column": layout.id_column,
                                    "reasons": layout.reasons}
    if merged:
        d["templates_note"] = ("Each template stands for several numeric columns whose names differ only in their "
                               "digits (shown as '#'), with statistics over all of them.")
    return d


def show(d: dict, max_chars: int | None = 4000) -> None:
    """Print a digest as indented JSON, and its size."""
    text = json.dumps(d, indent=1, ensure_ascii=False)
    print(text if max_chars is None or len(text) <= max_chars else text[:max_chars] + "\n... (cut here for display)")
    print(f"\n{len(text):,} characters: {len(d['templates'])} template(s), {len(d['columns'])} single column(s).")
