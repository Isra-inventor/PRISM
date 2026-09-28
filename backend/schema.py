"""Vocabularies and definitions: the single source of truth for PRISM Step 0.

Served at GET /api/vocabulary. The frontend builds every dropdown and tooltip
from it, and the AI prompt and response schema are generated from it. Adding a
new kind is a one-line edit in VOCABULARY (plus, optionally, a definition).
"""

SCHEMA_VERSION = "0.2"
PROMPT_VERSION = "step0-v2.0"

VOCABULARY = {
    "layout": ["samples_in_columns", "samples_in_rows", "long"],
    "omics_type": ["proteomics", "metabolomics", "lipidomics", "transcriptomics", "other", "unknown"],
    "supported_downstream": ["proteomics", "metabolomics"],
    "column_role": ["feature_id", "feature_annotation", "value", "sample_id", "sample_metadata",
                    "ignore", "unresolved"],
    "feature_annotation_kind": [
        "protein_accession", "gene_symbol", "protein_name", "peptide_sequence",
        "peptide_count", "sequence_coverage", "identification_score",
        "mz", "retention_time", "metabolite_name", "metabolite_db_id",
        "molecular_formula", "adduct",
        "flag_decoy", "flag_contaminant", "flag_other",
        "other_annotation",
    ],
    "sample_metadata_kind": [
        "subject_id", "timepoint", "batch", "run_order", "plate_or_slide",
        "group", "covariate_numeric", "covariate_categorical",
        "technical_replicate", "technical_numeric", "sample_type",
        "other_sample_metadata",
    ],
    "measurement_type": [
        "intensity", "count", "ratio", "log_ratio",
        "proportion_or_relative_abundance", "concentration",
        "normalized_abundance", "other_quantitative", "unknown",
    ],
    "scale": ["linear", "log2", "log10", "ln", "unknown"],
    "sample_type": ["study", "qc", "pool", "blank", "calibrator", "unknown"],
    "block_role": ["primary", "auxiliary", "excluded"],
    "yes_no_unsure": ["yes", "no", "not_sure"],
    "timepoint_detail": ["numeric", "ordinal_label", "date"],
    "provenance": ["signature", "computed", "ai_proposed_confirmed", "ai_proposed_corrected", "user_set"],
}

DEFINITIONS = {
    # roles
    "feature_id": "Uniquely identifies each feature (protein, peptide, metabolite).",
    "feature_annotation": "Describes a feature (name, gene, m/z, score, flag). One value per feature.",
    "value": "A measured quantity: one number per feature and sample.",
    "sample_id": "Identifies each sample (one row per sample in a samples-in-rows table).",
    "sample_metadata": "Describes a sample (subject, time point, batch, group, covariate...).",
    "ignore": "Not used for anything. The column is kept in the log but not in the outputs.",
    "unresolved": "Not decided yet. Must be resolved before finishing.",
    # annotation kinds
    "flag_decoy": "Marks rows that are decoy (reverse-database) hits, e.g. MaxQuant 'Reverse'.",
    "flag_contaminant": "Marks rows that are likely contaminants, e.g. MaxQuant 'Potential contaminant'.",
    "flag_other": "Marks rows as otherwise suspect, e.g. MaxQuant 'Only identified by site'.",
    "identification_score": "Score, q-value or PEP of the identification.",
    "metabolite_db_id": "Database identifier such as HMDB, KEGG, ChEBI or PubChem.",
    # sample metadata kinds
    "group": ("Categorical variable that plausibly defines comparison groups (case/control, "
              "treatment arm, severity). A candidate only: PRISM never decides which variable "
              "is the research outcome."),
    "technical_numeric": ("Instrument- or processing-derived numeric column (scale factors, "
                          "quality metrics). Not a feature."),
    "run_order": "Injection or acquisition order.",
    "timepoint": "Visit, day or time of sampling. Needs a detail: numeric, ordinal label or date.",
    "covariate_numeric": "Numeric sample characteristic, e.g. age, BMI, CD4 count, serum iron.",
    "covariate_categorical": "Categorical sample characteristic that is not a comparison group, e.g. sex.",
    "sample_type": "Study sample, QC, pool, blank or calibrator.",
    # measurement
    "intensity": "Signal intensity / peak area / abundance from the instrument.",
    "proportion_or_relative_abundance": "Values that are fractions of a total (0-1 or 0-100 %).",
    "normalized_abundance": "Values already normalized by the software.",
    # block roles
    "primary": "The main measurement matrix for this omics type (at most one per omics type).",
    "auxiliary": "Kept alongside, but not used as the main matrix (e.g. iBAQ next to LFQ).",
    "excluded": "Not used and not exported.",
    # layouts
    "samples_in_columns": "Each row is one feature; each sample has its own column.",
    "samples_in_rows": "Each row is one sample; each feature has its own column.",
    "long": "Each row is one (feature, sample) pair with a single value column.",
}

ALL_KINDS = VOCABULARY["feature_annotation_kind"] + VOCABULARY["sample_metadata_kind"]
UNRESOLVED = "unresolved"

HISTORY_QUESTIONS = [
    ("normalized", "Were the values normalized before upload?"),
    ("log_transformed", "Were the values log-transformed before upload?"),
    ("imputed", "Were missing values imputed (filled in) before upload?"),
    ("batch_corrected", "Was a batch correction applied before upload?"),
    ("features_or_samples_removed_before_upload", "Were any features or samples removed before upload?"),
]

UNSUPPORTED_NOTICE = "Recognized, but not yet supported by PRISM's downstream audit."


def vocabulary_payload():
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "vocabulary": VOCABULARY,
        "definitions": DEFINITIONS,
        "history_questions": [{"id": k, "question": q} for k, q in HISTORY_QUESTIONS],
        "unsupported_notice": UNSUPPORTED_NOTICE,
    }


def kinds_for_role(role):
    if role == "feature_annotation":
        return VOCABULARY["feature_annotation_kind"]
    if role == "sample_metadata":
        return VOCABULARY["sample_metadata_kind"]
    return []
