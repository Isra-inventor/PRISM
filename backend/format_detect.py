"""Deterministic format detection (PRISM Step 0).

A signature matches only if ALL of its required columns are present in the
header. No partial-credit scoring, no fuzzy matching. When a signature
matches, the value columns and the feature-ID column are resolved by fixed
rules, so the AI fallback is never called for that file.

Nothing in this module modifies data values: it only reads the header (and,
for the metabolomics feature table, checks whether sample cells *look*
numeric in order to label columns).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SIGNATURES = {
    "maxquant_proteinGroups": {
        "omics_type": "proteomics",
        "required_columns": {"Protein IDs", "Reverse"},
        "value_prefix": "LFQ intensity ",
        "feature_id_column": "Protein IDs",
    },
    "diann_pg_matrix": {
        "omics_type": "proteomics",
        "required_columns": {"PG.ProteinGroups"},
        "value_prefix": None,  # sample columns = everything after fixed PG.* columns
        "feature_id_column": "PG.ProteinGroups",
    },
    "spectronaut_results": {
        "omics_type": "proteomics",
        "required_columns": {"PEP.StrippedSequence", "F.PeakArea"},
        "value_prefix": "F.PeakArea",
        "feature_id_column": "PEP.StrippedSequence",
    },
    "fragpipe_combined_proteins": {
        "omics_type": "proteomics",
        "required_columns": {"Protein", "Indistinguishable Proteins"},
        "value_prefix": None,
        "feature_id_column": "Protein",
    },
    "generic_feature_table": {
        # XCMS/MZmine-style metabolomics: mz, rt, then one column per sample
        "omics_type": "metabolomics",
        "required_columns": {"mz", "rt"},
        "value_prefix": None,
        "feature_id_column": None,  # constructed from mz+rt
    },
}

PLATFORM_LABELS = {
    "maxquant_proteinGroups": "MaxQuant proteinGroups.txt",
    "diann_pg_matrix": "DIA-NN pg_matrix",
    "spectronaut_results": "Spectronaut report",
    "fragpipe_combined_proteins": "FragPipe combined_protein.tsv",
    "generic_feature_table": "Generic mz/rt feature table (XCMS / MZmine style)",
}

# FragPipe writes several per-sample quantities; pick one value type by a
# fixed priority order (first suffix that exists wins).
FRAGPIPE_VALUE_SUFFIXES = [" MaxLFQ Intensity", " Intensity", " Spectral Count"]
FRAGPIPE_EXCLUDED_INTENSITY = (" Unique Intensity", " Razor Intensity", " Total Intensity")
FRAGPIPE_EXCLUDED_COUNT = (" Unique Spectral Count", " Total Spectral Count", "Combined Spectral Count")

# Known per-feature annotation columns in XCMS / MZmine exports.
METABOLOMICS_ANNOTATION_COLUMNS = {
    "mz", "rt", "mzmin", "mzmax", "rtmin", "rtmax", "npeaks", "name", "id",
    "row id", "feature", "feature_id", "isotopes", "adduct", "pcgroup",
    "mzmed", "rtmed", "compound", "formula", "annotation", "ms2", "charge",
}

_NUMERIC_RE = re.compile(r"^[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?$")
_MISSING_TOKENS = {"", "na", "nan", "n/a", "null", "none", "-", "filtered", "#n/a"}


def looks_numeric(value: str) -> bool:
    return bool(_NUMERIC_RE.match(value.strip()))


def is_missing_token(value: str) -> bool:
    return value.strip().lower() in _MISSING_TOKENS


@dataclass
class DetectionResult:
    signature: str
    platform: str
    omics_type: str
    feature_id_column: str | None
    feature_id_note: str
    value_column_indices: list[int]
    roles: list[dict]  # one {index, column, role, reason} per column
    layout: str
    warnings: list[str] = field(default_factory=list)
    also_matched: list[str] = field(default_factory=list)


def match_signatures(header: list[str]) -> list[str]:
    """Return the names of all signatures whose required columns are ALL present."""
    present = set(header)
    return [name for name, sig in SIGNATURES.items() if sig["required_columns"] <= present]


def _value_indices(name: str, header: list[str], rows: list[list[str]]) -> tuple[list[int], list[str]]:
    sig = SIGNATURES[name]
    warnings: list[str] = []

    if sig["value_prefix"] is not None:
        prefix = sig["value_prefix"]
        idx = [i for i, c in enumerate(header) if c.startswith(prefix)]
        if not idx:
            warnings.append(f"No columns start with '{prefix}'; no value columns could be resolved.")
        return idx, warnings

    if name == "diann_pg_matrix":
        pg_idx = [i for i, c in enumerate(header) if c.startswith("PG.")]
        last = max(pg_idx)
        idx = list(range(last + 1, len(header)))
        if not idx:
            warnings.append("No columns follow the fixed PG.* columns; no sample columns found.")
        return idx, warnings

    if name == "fragpipe_combined_proteins":
        for suffix in FRAGPIPE_VALUE_SUFFIXES:
            idx = [
                i for i, c in enumerate(header)
                if c.endswith(suffix)
                and not c.endswith(FRAGPIPE_EXCLUDED_INTENSITY)
                and not c.endswith(FRAGPIPE_EXCLUDED_COUNT)
                and not c.startswith("Combined ")
            ]
            if idx:
                return idx, warnings
        warnings.append("No per-sample MaxLFQ Intensity / Intensity / Spectral Count columns found.")
        return [], warnings

    if name == "generic_feature_table":
        # Sample columns: not a known annotation column and numeric-looking
        # in the inspected rows (missing tokens are ignored).
        idx = []
        for i, c in enumerate(header):
            if c.strip().lower() in METABOLOMICS_ANNOTATION_COLUMNS:
                continue
            cells = [r[i] for r in rows if i < len(r) and not is_missing_token(r[i])]
            if cells and sum(looks_numeric(v) for v in cells) / len(cells) >= 0.9:
                idx.append(i)
        if not idx:
            warnings.append("No numeric sample columns found next to mz/rt.")
        return idx, warnings

    return [], warnings


def detect(header: list[str], rows: list[list[str]]) -> DetectionResult | None:
    """Deterministic detection. Returns None if no signature matches."""
    matched = match_signatures(header)
    if not matched:
        return None

    # Signatures are checked in declaration order; the generic mz/rt table is
    # last so a vendor-specific signature always takes precedence.
    name = matched[0]
    sig = SIGNATURES[name]
    value_idx, warnings = _value_indices(name, header, rows[:200])
    value_set = set(value_idx)

    feature_id_col = sig["feature_id_column"]
    feature_id_note = (
        f"'{feature_id_col}' column" if feature_id_col
        else "constructed from mz + rt (label only; no values changed)"
    )

    layout = "features_as_rows"
    long_sample_col = None
    if name == "spectronaut_results" and "R.FileName" in header:
        # Spectronaut 'normal' reports are long: one row per precursor per run.
        layout = "long"
        long_sample_col = "R.FileName"

    roles = []
    for i, col in enumerate(header):
        if i in value_set:
            role, reason = "feature_value", "Matches the signature's value-column rule"
        elif col == feature_id_col or (feature_id_col is None and col in ("mz", "rt")):
            role, reason = "feature_annotation", "Feature identifier"
        elif col == long_sample_col:
            role, reason = "sample_id", "Run / file identifier in a long-format report"
        elif name == "spectronaut_results" and col == "R.Condition":
            role, reason = "group_or_outcome", "Spectronaut condition label"
        else:
            role, reason = "feature_annotation", "Other column of the known format"
        roles.append({"index": i, "column": col, "role": role, "reason": reason})

    if len(matched) > 1:
        warnings.append(
            "Header also satisfies: " + ", ".join(PLATFORM_LABELS[m] for m in matched[1:])
            + f". Using {PLATFORM_LABELS[name]} (checked first)."
        )

    return DetectionResult(
        signature=name,
        platform=PLATFORM_LABELS[name],
        omics_type=sig["omics_type"],
        feature_id_column=feature_id_col,
        feature_id_note=feature_id_note,
        value_column_indices=value_idx,
        roles=roles,
        layout=layout,
        warnings=warnings,
        also_matched=matched[1:],
    )
