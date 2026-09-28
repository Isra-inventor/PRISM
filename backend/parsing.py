"""Parsing (spec 3.1): read a delimited table as strings and report, never fix.

Every cell stays the exact string from the file. Oddities (duplicate or empty
headers, empty rows/columns, decimal commas, thousands separators, padding,
missing-value tokens) are detected and returned in the parse report.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from collections import Counter, OrderedDict
from pathlib import Path

ACCEPTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".tab"}
DELIMITERS = [",", "\t", ";", "|"]
MISSING_TOKENS = ["", "NA", "NaN", "N/A", "#N/A", "NULL", "-", "Filtered"]
_MISSING_LOWER = {t.lower() for t in MISSING_TOKENS} | {"nan", "null", "none", "n/a", "#n/a", "na"}
WRONG_FORMAT_MESSAGE = (
    "PRISM accepts quantified tables only, as CSV or TSV (comma-, tab-, semicolon- or "
    "pipe-separated .csv / .tsv / .txt). Raw spectra and vendor or binary files (.raw, .d, "
    ".wiff, .mzML, .mzXML, .xlsx, ...) are not processed. Export a quantified protein, "
    "peptide or feature table from your analysis software and upload that."
)

_NUM_DOT = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_NUM_COMMA_DECIMAL = re.compile(r"^[+-]?\d+,\d+([eE][+-]?\d+)?$")
_NUM_THOUSANDS_COMMA = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")
_NUM_THOUSANDS_DOT = re.compile(r"^[+-]?\d{1,3}(\.\d{3})+(,\d+)?$")
_UNNAMED = re.compile(r"^(Unnamed(: ?\d+)?|V\d+|X\d*|\.\.\.\d+)$")


class InputError(Exception):
    pass


def sanitize_filename(name):
    base = Path(str(name or "upload")).name
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "upload"
    return base[:100]


def is_missing(value):
    return value.strip().lower() in _MISSING_LOWER


def missing_token_label(value):
    v = value.strip()
    return "(empty)" if v == "" else v


def parse_number(value, decimal_comma=False):
    """Float for a numeric-looking cell, else None. Missing tokens -> None.
    Only used for statistics: the source string is never changed."""
    v = value.strip()
    if not v or v.lower() in _MISSING_LOWER:
        return None
    if _NUM_DOT.match(v):
        return float(v)
    if _NUM_THOUSANDS_COMMA.match(v) and not decimal_comma:
        return float(v.replace(",", ""))
    if decimal_comma and _NUM_COMMA_DECIMAL.match(v):
        return float(v.replace(",", "."))
    if decimal_comma and _NUM_THOUSANDS_DOT.match(v):
        return float(v.replace(".", "").replace(",", "."))
    if v.lower() in ("inf", "+inf", "-inf", "infinity", "-infinity"):
        return float(v.lower().replace("infinity", "inf"))
    return None


def _decode(raw):
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8"), "utf-8", True
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16"), "utf-16", True
    try:
        return raw.decode("utf-8"), "utf-8", False
    except UnicodeDecodeError:
        return raw.decode("latin-1"), "latin-1", False


def sniff_delimiter(lines, ext):
    """The candidate that splits the most sampled lines into the same (>1) number of fields."""
    lines = [l for l in lines if l.strip()][:30]
    best, best_score = None, -1
    for d in DELIMITERS:
        counts = [len(next(csv.reader([l], delimiter=d))) for l in lines]
        if not counts or counts[0] < 2:
            continue
        consistent = sum(1 for c in counts if c == counts[0])
        score = consistent * 10000 + counts[0] + (1 if (d == "\t" and ext in (".tsv", ".txt", ".tab")) else 0)
        if score > best_score:
            best, best_score = d, score
    return best


def parse_bytes(filename, raw):
    """Parse an uploaded file. Returns a dict with header, rows (lists of the
    original strings), delimiter, encoding, sha256 and the parse report."""
    ext = Path(filename).suffix.lower()
    if ext not in ACCEPTED_EXTENSIONS:
        raise InputError(f"'{filename}' is not a CSV/TSV file. " + WRONG_FORMAT_MESSAGE)
    if not raw.strip():
        raise InputError("The file is empty.")
    if b"\x00" in raw[:8192] and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise InputError(f"'{filename}' looks like a binary file. " + WRONG_FORMAT_MESSAGE)

    text, encoding, bom = _decode(raw)
    lines = text.splitlines()
    delimiter = sniff_delimiter(lines, ext)
    if delimiter is None:
        raise InputError("Only one column was found; no comma, tab, semicolon or pipe separator. "
                         + WRONG_FORMAT_MESSAGE)

    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    header = next(reader)
    rows, empty_rows = [], 0
    for r in reader:
        if not any(c.strip() for c in r):
            empty_rows += 1
            continue
        rows.append(r)

    n_cols = len(header)
    ragged = [i + 2 for i, r in enumerate(rows) if len(r) != n_cols]
    report = build_report(header, rows, delimiter, encoding, bom, empty_rows, ragged)
    return {
        "header": header,
        "rows": rows,
        "delimiter": delimiter,
        "encoding": encoding,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "parse_report": report,
    }


def cell(row, i):
    return row[i] if i < len(row) else ""


def build_report(header, rows, delimiter, encoding, bom, empty_rows, ragged):
    n_cols = len(header)
    dup = OrderedDict()
    for i, h in enumerate(header):
        dup.setdefault(h, []).append(i)
    duplicates = [{"name": h, "positions": [p + 1 for p in pos]} for h, pos in dup.items() if len(pos) > 1]
    empty_headers = [i + 1 for i, h in enumerate(header) if not h.strip() or _UNNAMED.match(h.strip())]

    labels = column_labels(header)
    missing_tokens = Counter()
    empty_columns, decimal_comma_cols, thousands_cols, padded_cols = [], [], [], []
    for i in range(n_cols):
        values = [cell(r, i) for r in rows]
        present = []
        padded = 0
        for v in values:
            if is_missing(v):
                missing_tokens[missing_token_label(v)] += 1
            else:
                present.append(v)
                if v != v.strip():
                    padded += 1
        name = labels[i]
        if not present:
            empty_columns.append(name)
            continue
        if padded:
            padded_cols.append({"column": name, "cells": padded})
        if delimiter != ",":
            n_dc = sum(1 for v in present if _NUM_COMMA_DECIMAL.match(v.strip()))
            if n_dc and n_dc >= 0.5 * len(present):
                decimal_comma_cols.append(name)
        n_th = sum(1 for v in present if _NUM_THOUSANDS_COMMA.match(v.strip()))
        if n_th and name not in decimal_comma_cols:
            thousands_cols.append({"column": name, "cells": n_th})

    return {
        "delimiter": {"\t": "tab", ",": "comma", ";": "semicolon", "|": "pipe"}[delimiter],
        "encoding": encoding,
        "bom": bom,
        "n_rows": len(rows),
        "n_columns": n_cols,
        "duplicate_column_names": duplicates,
        "empty_or_unnamed_headers": empty_headers,
        "fully_empty_rows": empty_rows,
        "fully_empty_columns": empty_columns,
        "ragged_rows": {"count": len(ragged), "first_lines": ragged[:10]},
        "decimal_comma_columns": decimal_comma_cols,
        "thousands_separator_columns": thousands_cols,
        "whitespace_padded_columns": padded_cols,
        "missing_value_tokens": dict(missing_tokens.most_common()),
        "note": ("All values are kept exactly as in the file. Missing-value tokens are treated as "
                 "missing; 0 is never treated as missing. Fully empty lines are not counted as rows."),
    }


def column_labels(header):
    """Unique, human-readable label per column (duplicates and blanks disambiguated)."""
    counts = Counter(header)
    out = []
    for i, name in enumerate(header):
        if not name.strip():
            out.append(f"(unnamed column {i + 1})")
        elif counts[name] > 1:
            out.append(f"{name} [column {i + 1}]")
        else:
            out.append(name)
    return out
