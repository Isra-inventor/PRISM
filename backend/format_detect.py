"""Deterministic format detection (spec 3.5, kept from v1).

A signature matches only if ALL of its required columns are present in the
header (no fuzzy matching). A match pre-fills the proposal with provenance
'signature'. Grouping and profiling still run: e.g. MaxQuant has several
per-sample families; each is described, none is ranked.
"""

from __future__ import annotations

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

SOFTWARE = {
    "maxquant_proteinGroups": "MaxQuant",
    "diann_pg_matrix": "DIA-NN",
    "spectronaut_results": "Spectronaut",
    "fragpipe_combined_proteins": "FragPipe",
    "generic_feature_table": "XCMS / MZmine-style",
}

PLATFORM_LABELS = {
    "maxquant_proteinGroups": "MaxQuant proteinGroups.txt",
    "diann_pg_matrix": "DIA-NN pg_matrix",
    "spectronaut_results": "Spectronaut report",
    "fragpipe_combined_proteins": "FragPipe combined_protein.tsv",
    "generic_feature_table": "Generic mz/rt feature table (XCMS / MZmine style)",
}

# Known single columns per signature: column name -> (role, label, extra fields)
_SUSPECT = {"marks_rows_as_suspect": True}
KNOWN_COLUMNS = {
    "maxquant_proteinGroups": {
        "Protein IDs": ("feature_id", "protein group accessions", {}),
        "Majority protein IDs": ("feature_annotation", "majority protein accessions", {}),
        "Protein names": ("feature_annotation", "protein names", {}),
        "Gene names": ("feature_annotation", "gene symbol", {}),
        "Fasta headers": ("feature_annotation", "FASTA header", {}),
        "Peptides": ("feature_annotation", "number of peptides", {}),
        "Razor + unique peptides": ("feature_annotation", "number of razor + unique peptides", {}),
        "Unique peptides": ("feature_annotation", "number of unique peptides", {}),
        "Sequence coverage [%]": ("feature_annotation", "sequence coverage (%)", {}),
        "Q-value": ("feature_annotation", "identification q-value", {}),
        "Score": ("feature_annotation", "identification score", {}),
        "Mol. weight [kDa]": ("feature_annotation", "molecular weight (kDa)", {}),
        "Intensity": ("feature_annotation", "summed intensity over all samples", {}),
        "iBAQ": ("feature_annotation", "summed iBAQ over all samples", {}),
        "Only identified by site": ("feature_annotation", "only identified by a modification site flag", _SUSPECT),
        "Reverse": ("feature_annotation", "decoy (reverse database) hit flag", _SUSPECT),
        "Potential contaminant": ("feature_annotation", "potential contaminant flag", _SUSPECT),
        "id": ("feature_annotation", "MaxQuant row id", {}),
    },
    "diann_pg_matrix": {
        "PG.ProteinGroups": ("feature_id", "protein group accessions", {}),
        "PG.ProteinAccessions": ("feature_annotation", "protein accessions", {}),
        "PG.ProteinNames": ("feature_annotation", "protein names", {}),
        "PG.Genes": ("feature_annotation", "gene symbol", {}),
        "PG.Q.Value": ("feature_annotation", "protein group q-value", {}),
    },
    "spectronaut_results": {
        "PEP.StrippedSequence": ("feature_id", "peptide sequence", {}),
        "R.FileName": ("sample_id", "raw file name", {}),
        "R.Condition": ("sample_metadata", "condition", {"audit_kind": "group"}),
        "R.Replicate": ("sample_metadata", "replicate number", {"audit_kind": "technical_replicate"}),
        "F.PeakArea": ("value", "fragment peak area", {}),
        "PG.ProteinGroups": ("feature_annotation", "protein group accessions", {}),
        "PG.Genes": ("feature_annotation", "gene symbol", {}),
    },
    "fragpipe_combined_proteins": {
        "Protein": ("feature_id", "protein", {}),
        "Protein ID": ("feature_annotation", "protein accession", {}),
        "Entry Name": ("feature_annotation", "UniProt entry name", {}),
        "Gene": ("feature_annotation", "gene symbol", {}),
        "Description": ("feature_annotation", "protein description", {}),
        "Protein Probability": ("feature_annotation", "protein probability", {}),
        "Top Peptide Probability": ("feature_annotation", "top peptide probability", {}),
        "Indistinguishable Proteins": ("feature_annotation", "indistinguishable proteins", {}),
    },
    "generic_feature_table": {
        "mz": ("feature_annotation", "m/z", {}),
        "rt": ("feature_annotation", "retention time", {}),
    },
}

# Per-sample column families: pattern text -> label
KNOWN_FAMILIES = {
    "maxquant_proteinGroups": {
        "LFQ intensity ": "LFQ intensity",
        "Intensity ": "raw intensity",
        "iBAQ ": "iBAQ intensity",
        "Peptides ": "peptide count per sample",
        "Razor + unique peptides ": "razor + unique peptide count per sample",
        "Unique peptides ": "unique peptide count per sample",
        "MS/MS count ": "MS/MS count per sample",
        "Sequence coverage ": "sequence coverage per sample (%)",
    },
    "fragpipe_combined_proteins": {
        " MaxLFQ Intensity": "MaxLFQ intensity",
        " Intensity": "intensity",
        " Unique Intensity": "unique intensity",
        " Razor Intensity": "razor intensity",
        " Spectral Count": "spectral count",
        " Unique Spectral Count": "unique spectral count",
        " Total Spectral Count": "total spectral count",
    },
}


def match_signatures(header):
    """Names of all signatures whose required columns are ALL present."""
    present = set(header)
    return [name for name, sig in SIGNATURES.items() if sig["required_columns"] <= present]


def signature_hint(header):
    """One sentence for the AI digest, or None. Signatures are frozen: hint only."""
    matched = match_signatures(header)
    if not matched:
        return None
    return f"headers match the known {PLATFORM_LABELS[matched[0]]} pattern"


def _family_of(name, families):
    """Longest known family pattern this column name carries (exact text; prefix
    patterns end with a space, suffix patterns start with one)."""
    best = None
    for pat in families:
        hit = name.startswith(pat) if pat.endswith(" ") else name.endswith(pat)
        if hit and len(name) > len(pat) and (best is None or len(pat) > len(best)):
            best = pat
    return best


def signature_prefill(header, cols):
    """Starting point for MANUAL mode (AI off or unavailable), or None. Same shape as
    an AI proposal: groups are pid -> {indices, item, origin}. The groups come from
    the known format's exact column names; every other column is its own unresolved
    group. Items carry source 'computed'; the user confirms everything."""
    from .grouping import unmentioned_item
    matched = match_signatures(header)
    if not matched:
        return None
    name = matched[0]
    sig = SIGNATURES[name]
    ev = f"Headers match the known {PLATFORM_LABELS[name]} pattern ({', '.join(sorted(sig['required_columns']))})."
    known = KNOWN_COLUMNS.get(name, {})
    families = KNOWN_FAMILIES.get(name, {})
    assay = f"{sig['omics_type']} ({SOFTWARE[name]})"
    groups, fam_pid = {}, {}
    item = lambda **k: dict({"confidence": 0.9, "source": "computed", "audit_kind": None, "assay_label": None,
                             "marks_rows_as_suspect": False}, **k)
    pg_pos = max((i for i, h in enumerate(header) if h.startswith("PG.")), default=-1)
    for i, col in enumerate(header):
        if col in known:
            role, label, extra = known[col]
            groups[f"k{i}"] = {"indices": [i], "origin": "signature", "item": item(
                role=role, label=label, evidence=f"Known {SOFTWARE[name]} column '{col}'.",
                assay_label=assay if role == "value" else None, **extra)}
            continue
        pat = _family_of(col, families) if cols.is_numeric(i) else None
        if pat is None and name == "diann_pg_matrix" and cols.is_numeric(i) and i > pg_pos:
            pat = "__diann_samples__"
        if pat is None:
            groups[f"u{i}"] = {"indices": [i], "origin": "unmentioned", "item": unmentioned_item()}
            continue
        if pat not in fam_pid:
            fam_pid[pat] = f"fam{len(fam_pid) + 1}"
            label = families.get(pat, "protein group quantity")
            groups[fam_pid[pat]] = {"indices": [], "origin": "signature", "item": item(
                role="value", label=label, assay_label=assay,
                evidence=(f"{SOFTWARE[name]} per-sample column family '{pat.strip()} ...'." if pat in families else
                          "DIA-NN: sample columns are all columns after the fixed PG.* columns."))}
        groups[fam_pid[pat]]["indices"].append(i)
    prop = {
        "signature": name, "platform": PLATFORM_LABELS[name], "groups": groups,
        "layout": {"value": "long" if name == "spectronaut_results" else "samples_in_columns",
                   "confidence": 0.9, "evidence": ev, "source": "computed"},
        "assays": [{"assay_label": assay, "omics_type": sig["omics_type"], "source_software": SOFTWARE[name],
                    "in_supported_scope": "yes", "scope_reason": "known quantified-omics export",
                    "confidence": 0.9, "evidence": ev, "source": "computed", "feature_pids": []}],
    }
    if sig["feature_id_column"] in header:
        prop["assays"][0]["feature_pids"] = [f"k{header.index(sig['feature_id_column'])}"]
    elif name == "generic_feature_table":
        prop["assays"][0]["feature_pids"] = [f"k{header.index('mz')}", f"k{header.index('rt')}"]
        prop["assays"][0]["evidence"] = ev + " Features are identified by m/z + retention time."
    return prop
