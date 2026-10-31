"""The closed lists: the only places where the AI (or you) must pick from fixed values.

Rule (kept from the old website): a list is closed where code branches on the value, and
free text everywhere else. Assay labels, omics type, software names and every group label are
free text, written by the AI or by you, and are never listed here.
"""

SCHEMA_VERSION = "0.4.1"
UNRESOLVED = "unresolved"

VOCABULARY = {
    "layout": ["samples_in_columns", "samples_in_rows", "long"],
    "column_role": ["feature_id", "feature_annotation", "value", "sample_id", "sample_metadata",
                    "ignore", UNRESOLVED],
    "audit_kind": ["subject_id", "timepoint", "batch", "run_order", "technical_replicate", "sample_type",
                   "group", "covariate", "other"],
    "in_supported_scope": ["yes", "no", "unsure"],
    "omics_family": ["proteomics", "metabolomics", "lipidomics", "transcriptomics", "epigenomics", "genomics",
                     "microbiome", "other", "unknown"],
    "provenance": ["computed", "ai_proposed_confirmed", "ai_proposed_corrected", "user_set"],
}

DEFINITIONS = {
    "feature_id": "Uniquely identifies each feature (protein, peptide, metabolite, ...).",
    "feature_annotation": "Describes a feature (name, gene, m/z, score, flag). One value per feature.",
    "value": "A measured quantity: one number per feature and sample.",
    "sample_id": "Identifies each sample (one row per sample in a samples-in-rows table).",
    "sample_metadata": "Describes a sample (subject, time point, batch, group, covariate ...).",
    "ignore": "Not used. Kept in the log, left out of the outputs.",
    "unresolved": "Not decided yet. Must be resolved before finishing.",
    "subject_id": "The individual a sample came from; repeats across repeated measures.",
    "timepoint": "When the sample was taken (visit, day, time).",
    "batch": "Processing / acquisition batch or plate.",
    "run_order": "Injection or acquisition order (drift).",
    "technical_replicate": "Marks technical replicates of the same sample.",
    "sample_type": "Says whether a sample is a study sample, QC, blank, pool or calibrator.",
    "group": "A categorical variable that may define comparison groups. A candidate only: PRISM never "
             "decides which variable is the research outcome.",
    "covariate": "Any other sample characteristic (age, sex, BMI, clinical values ...).",
    "other": "Sample information that fits none of the above.",
    "samples_in_columns": "Each row is one feature; each sample has its own column.",
    "samples_in_rows": "Each row is one sample; each feature has its own column.",
    "long": "Each row is one (feature, sample) pair with a single value column.",
}

HISTORY_QUESTIONS = [
    ("normalized", "Were the values normalized before upload?"),
    ("log_transformed", "Were the values log-transformed before upload?"),
    ("imputed", "Were missing values imputed (filled in) before upload?"),
    ("batch_corrected", "Was a batch correction applied before upload?"),
    ("features_or_samples_removed_before_upload", "Were any features or samples removed before upload?"),
]
