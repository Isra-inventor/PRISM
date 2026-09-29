"""Deterministic format detection (spec 3.5, kept from v1).

A signature matches only if ALL of its required columns are present in the
header (no fuzzy matching). A match pre-fills the proposal with provenance
'signature'. Grouping and profiling still run: e.g. MaxQuant has several
per-sample families and the wizard still asks which one is primary.
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
        "F.PeakArea": ("value", "fragment peak area", {"block_role": "primary"}),
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

# Per-sample column families: pattern text -> (block_role, label)
KNOWN_FAMILIES = {
    "maxquant_proteinGroups": {
        "LFQ intensity ": ("primary", "LFQ intensity"),
        "Intensity ": ("auxiliary", "raw intensity"),
        "iBAQ ": ("auxiliary", "iBAQ intensity"),
        "Peptides ": ("auxiliary", "peptide count per sample"),
        "Razor + unique peptides ": ("auxiliary", "razor + unique peptide count per sample"),
        "Unique peptides ": ("auxiliary", "unique peptide count per sample"),
        "MS/MS count ": ("auxiliary", "MS/MS count per sample"),
        "Sequence coverage ": ("auxiliary", "sequence coverage per sample (%)"),
    },
    "fragpipe_combined_proteins": {
        " MaxLFQ Intensity": ("primary", "MaxLFQ intensity"),
        " Intensity": ("auxiliary", "intensity"),
        " Unique Intensity": ("auxiliary", "unique intensity"),
        " Razor Intensity": ("auxiliary", "razor intensity"),
        " Spectral Count": ("auxiliary", "spectral count"),
        " Unique Spectral Count": ("auxiliary", "unique spectral count"),
        " Total Spectral Count": ("auxiliary", "total spectral count"),
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


def signature_prefill(header, groups):
    """Starting point for MANUAL mode (AI off or unavailable), or None.
    Items carry source 'computed'; the user confirms everything."""
    matched = match_signatures(header)
    if not matched:
        return None
    name = matched[0]
    sig = SIGNATURES[name]
    ev = f"Headers match the known {PLATFORM_LABELS[name]} pattern ({', '.join(sorted(sig['required_columns']))})."
    known = KNOWN_COLUMNS.get(name, {})
    families = KNOWN_FAMILIES.get(name, {})
    assay = f"{sig['omics_type']} ({SOFTWARE[name]})"
    prop = {
        "signature": name, "platform": PLATFORM_LABELS[name],
        "layout": {"value": "long" if name == "spectronaut_results" else "samples_in_columns",
                   "confidence": 0.9, "evidence": ev, "source": "computed"},
        "assays": [{"assay_label": assay, "omics_type": sig["omics_type"], "source_software": SOFTWARE[name],
                    "in_supported_scope": "yes", "scope_reason": "known quantified-omics export",
                    "confidence": 0.9, "evidence": ev, "source": "computed"}],
        "groups": {},
    }
    by_col = {c: g for g in groups for c in g["columns"]}
    if sig["feature_id_column"] and sig["feature_id_column"] in by_col:
        prop["feature_identity"] = {"group_ids": [by_col[sig["feature_id_column"]]["group_id"]], "composite": False,
                                    "confidence": 0.9, "evidence": ev, "source": "computed"}
    elif name == "generic_feature_table":
        prop["feature_identity"] = {"group_ids": [by_col["mz"]["group_id"], by_col["rt"]["group_id"]],
                                    "composite": True, "confidence": 0.9, "source": "computed",
                                    "evidence": ev + " Features are identified by m/z + retention time."}
    for g in groups:
        col = g["columns"][0]
        if g["n_columns"] == 1 and col in known:
            role, label, extra = known[col]
            item = {"role": role, "label": label, "confidence": 0.9, "source": "computed",
                    "evidence": f"Known {SOFTWARE[name]} column '{col}'."}
            item.update(extra)
            if role == "value":
                item["assay_label"] = assay
            prop["groups"][g["group_id"]] = item
            continue
        pat = (g.get("pattern") or {}).get("text")
        if pat in families:
            block_role, label = families[pat]
            prop["groups"][g["group_id"]] = {
                "role": "value", "label": label, "block_role": block_role, "assay_label": assay,
                "confidence": 0.9, "source": "computed",
                "evidence": f"{SOFTWARE[name]} per-sample column family '{pat.strip()} ...'."}
    if name == "diann_pg_matrix":
        pg_pos = max(i for i, h in enumerate(header) if h.startswith("PG."))
        for g in groups:
            if g["type"] == "numeric" and min(g["indices"]) > pg_pos and g["group_id"] not in prop["groups"]:
                prop["groups"][g["group_id"]] = {
                    "role": "value", "label": "protein group quantity", "block_role": "primary",
                    "assay_label": assay, "confidence": 0.9, "source": "computed",
                    "evidence": "DIA-NN: sample columns are all columns after the fixed PG.* columns."}
    return prop
