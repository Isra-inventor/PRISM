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

# Known single columns per signature: column name -> (role, kind)
KNOWN_COLUMNS = {
    "maxquant_proteinGroups": {
        "Protein IDs": ("feature_id", "protein_accession"),
        "Majority protein IDs": ("feature_annotation", "protein_accession"),
        "Protein names": ("feature_annotation", "protein_name"),
        "Gene names": ("feature_annotation", "gene_symbol"),
        "Fasta headers": ("feature_annotation", "other_annotation"),
        "Peptides": ("feature_annotation", "peptide_count"),
        "Razor + unique peptides": ("feature_annotation", "peptide_count"),
        "Unique peptides": ("feature_annotation", "peptide_count"),
        "Sequence coverage [%]": ("feature_annotation", "sequence_coverage"),
        "Q-value": ("feature_annotation", "identification_score"),
        "Score": ("feature_annotation", "identification_score"),
        "Mol. weight [kDa]": ("feature_annotation", "other_annotation"),
        "Intensity": ("feature_annotation", "other_annotation"),
        "iBAQ": ("feature_annotation", "other_annotation"),
        "Only identified by site": ("feature_annotation", "flag_other"),
        "Reverse": ("feature_annotation", "flag_decoy"),
        "Potential contaminant": ("feature_annotation", "flag_contaminant"),
        "id": ("feature_annotation", "other_annotation"),
    },
    "diann_pg_matrix": {
        "PG.ProteinGroups": ("feature_id", "protein_accession"),
        "PG.ProteinAccessions": ("feature_annotation", "protein_accession"),
        "PG.ProteinNames": ("feature_annotation", "protein_name"),
        "PG.Genes": ("feature_annotation", "gene_symbol"),
        "PG.Q.Value": ("feature_annotation", "identification_score"),
    },
    "spectronaut_results": {
        "PEP.StrippedSequence": ("feature_id", "peptide_sequence"),
        "R.FileName": ("sample_id", None),
        "R.Condition": ("sample_metadata", "group"),
        "R.Replicate": ("sample_metadata", "technical_replicate"),
        "F.PeakArea": ("value", None),
        "PG.ProteinGroups": ("feature_annotation", "protein_accession"),
        "PG.Genes": ("feature_annotation", "gene_symbol"),
    },
    "fragpipe_combined_proteins": {
        "Protein": ("feature_id", "protein_accession"),
        "Protein ID": ("feature_annotation", "protein_accession"),
        "Entry Name": ("feature_annotation", "protein_name"),
        "Gene": ("feature_annotation", "gene_symbol"),
        "Description": ("feature_annotation", "protein_name"),
        "Protein Probability": ("feature_annotation", "identification_score"),
        "Top Peptide Probability": ("feature_annotation", "identification_score"),
        "Indistinguishable Proteins": ("feature_annotation", "other_annotation"),
    },
    "generic_feature_table": {
        "mz": ("feature_annotation", "mz"),
        "rt": ("feature_annotation", "retention_time"),
    },
}

# Per-sample column families: pattern text -> (block_role, measurement_type, label)
KNOWN_FAMILIES = {
    "maxquant_proteinGroups": {
        "LFQ intensity ": ("primary", "intensity", "LFQ intensity"),
        "Intensity ": ("auxiliary", "intensity", "Intensity"),
        "iBAQ ": ("auxiliary", "intensity", "iBAQ"),
        "Peptides ": ("auxiliary", "count", "Peptides per sample"),
        "Razor + unique peptides ": ("auxiliary", "count", "Razor + unique peptides per sample"),
        "Unique peptides ": ("auxiliary", "count", "Unique peptides per sample"),
        "MS/MS count ": ("auxiliary", "count", "MS/MS count"),
        "Sequence coverage ": ("auxiliary", "other_quantitative", "Sequence coverage per sample"),
    },
    "fragpipe_combined_proteins": {
        " MaxLFQ Intensity": ("primary", "intensity", "MaxLFQ intensity"),
        " Intensity": ("auxiliary", "intensity", "Intensity"),
        " Unique Intensity": ("auxiliary", "intensity", "Unique intensity"),
        " Razor Intensity": ("auxiliary", "intensity", "Razor intensity"),
        " Spectral Count": ("auxiliary", "count", "Spectral count"),
        " Unique Spectral Count": ("auxiliary", "count", "Unique spectral count"),
        " Total Spectral Count": ("auxiliary", "count", "Total spectral count"),
    },
}


def match_signatures(header):
    """Names of all signatures whose required columns are ALL present."""
    present = set(header)
    return [name for name, sig in SIGNATURES.items() if sig["required_columns"] <= present]


def _item(value, evidence, **extra):
    d = {"value": value, "confidence": 1.0, "evidence": evidence, "source": "signature"}
    d.update(extra)
    return d


def signature_prefill(header, groups):
    """Proposal fragment from the first matching signature, or None.

    Only facts the signature knows for certain are filled; everything else is
    left for the AI (or the user)."""
    matched = match_signatures(header)
    if not matched:
        return None
    name = matched[0]
    sig = SIGNATURES[name]
    ev = f"Matches the {PLATFORM_LABELS[name]} signature (columns: {', '.join(sorted(sig['required_columns']))})."
    known = KNOWN_COLUMNS.get(name, {})
    families = KNOWN_FAMILIES.get(name, {})
    long_format = name == "spectronaut_results"
    prop = {
        "signature": name,
        "platform": PLATFORM_LABELS[name],
        "also_matched": matched[1:],
        "layout": _item("long" if long_format else "samples_in_columns", ev),
        "omics_type": _item(sig["omics_type"], ev),
        "source_software": _item(SOFTWARE[name], ev),
        "groups": {},
    }
    by_col = {}
    for g in groups:
        for c in g["columns"]:
            by_col[c] = g
    # feature identity
    if sig["feature_id_column"] and sig["feature_id_column"] in by_col:
        prop["feature_identity"] = {"group_ids": [by_col[sig["feature_id_column"]]["group_id"]], "composite": False,
                                    "confidence": 1.0, "evidence": ev, "source": "signature"}
    elif name == "generic_feature_table":
        prop["feature_identity"] = {"group_ids": [by_col["mz"]["group_id"], by_col["rt"]["group_id"]],
                                    "composite": True, "confidence": 1.0,
                                    "evidence": ev + " Features are identified by m/z + retention time.",
                                    "source": "signature"}
    for g in groups:
        col = g["columns"][0]
        if g["n_columns"] == 1 and col in known:
            role, kind = known[col]
            prop["groups"][g["group_id"]] = {"role": role, "kind": kind, "confidence": 1.0,
                                             "evidence": f"Known {SOFTWARE[name]} column '{col}'.",
                                             "source": "signature"}
            if role == "value":
                prop["groups"][g["group_id"]].update(measurement_type="intensity", label=col,
                                                     block_role="primary", omics_type=sig["omics_type"])
            continue
        pat = (g.get("pattern") or {}).get("text")
        if pat in families:
            block_role, mtype, label = families[pat]
            prop["groups"][g["group_id"]] = {
                "role": "value", "measurement_type": mtype, "label": label, "block_role": block_role,
                "omics_type": sig["omics_type"], "confidence": 1.0, "source": "signature",
                "evidence": f"{SOFTWARE[name]} per-sample column family '{pat.strip()} ...'"
                            + (" (the signature's main value columns)." if block_role == "primary" else "."),
            }
    if name == "diann_pg_matrix":
        pg_pos = max(i for i, h in enumerate(header) if h.startswith("PG."))
        for g in groups:
            if g["type"] == "numeric" and min(g["indices"]) > pg_pos and g["group_id"] not in prop["groups"]:
                prop["groups"][g["group_id"]] = {
                    "role": "value", "measurement_type": "intensity", "label": "Protein group quantity",
                    "block_role": "primary", "omics_type": "proteomics", "confidence": 1.0, "source": "signature",
                    "evidence": "DIA-NN: sample columns are all columns after the fixed PG.* columns."}
    return prop
