"""The closed lists: the only places where the AI (or you) must pick from fixed values.

Rule (kept from the old website): a list is closed where code branches on the value, and
free text everywhere else. Assay labels, omics type, software names and every group label are
free text, written by the AI or by you, and are never listed here.
"""

SCHEMA_VERSION = "0.4.1"

# What PRISM's later steps (audit, preprocessing advice) support. Files outside it are still
# recognized and described, but their assay is marked in_supported_scope "no" or "unsure".
# Widening the scope later means editing this text.
SCOPE = ("Proteomics only: protein-level abundance tables, one value per protein and sample. "
         "Affinity platforms (SomaScan RFU, Olink NPX) and mass-spectrometry tool outputs (MaxQuant "
         "proteinGroups, DIA-NN pg_matrix, Spectronaut, FragPipe combined_protein), including paired "
         "and longitudinal designs. Peptide-, precursor- or PSM-level tables: 'unsure' (they need "
         "summarising to proteins first). Metabolomics, lipidomics, transcriptomics and other omics: 'no'.")
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
    "sample_type": ["study", "qc", "blank", "pool", "calibrator", "reference", "other"],
    "yes_no_unsure": ["yes", "no", "not_sure"],
    "time_unit": ["minutes", "hours", "days", "weeks", "months", "years", "visit_index"],
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
    "study": "A study sample (from a subject or experimental unit), as opposed to the types below.",
    "qc": "A quality-control sample (e.g. a pooled QC or a platform QC).",
    "blank": "A blank or buffer injection / well with no sample.",
    "pool": "A pool of study samples run for monitoring.",
    "calibrator": "A calibrator sample used by the platform for scaling.",
    "reference": "A reference or bridging sample shared across plates or batches.",
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
