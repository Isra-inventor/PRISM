"""Closed vocabularies: the single source of truth, served at GET /api/vocabulary.

Rule: closed where code branches on the value, open (free text) where the
value is descriptive. Only the sets below are closed. Omics type, software,
assay label, annotation / block / sample labels and details are free text
written by the AI (or the user) and are never enumerated here.
"""

from .config import SCOPE_DESCRIPTION

SCHEMA_VERSION = "0.2.1"
PROMPT_VERSION = "step0-v2.1"
UNRESOLVED = "unresolved"

VOCABULARY = {
    # canonicalization branches on it
    "layout": ["samples_in_columns", "samples_in_rows", "long"],
    # decides which output file a column goes to
    "column_role": ["feature_id", "feature_annotation", "value", "sample_id", "sample_metadata",
                    "ignore", "unresolved"],
    "block_role": ["primary", "auxiliary", "excluded"],
    # only for sample_metadata columns; the later audit reads exactly these ('other' is the escape hatch)
    "audit_kind": ["subject_id", "timepoint", "batch", "run_order", "technical_replicate", "sample_type",
                   "group", "covariate", "other"],
    "yes_no_unsure": ["yes", "no", "not_sure"],
    "in_supported_scope": ["yes", "no", "unsure"],
    "provenance": ["computed", "ai_proposed_confirmed", "ai_proposed_corrected", "user_set"],
    "booleans": ["keep", "marks_rows_as_suspect", "is_study_sample"],
}

DEFINITIONS = {
    "feature_id": "Uniquely identifies each feature (protein, peptide, metabolite, ...).",
    "feature_annotation": "Describes a feature (name, gene, m/z, score, flag). One value per feature.",
    "value": "A measured quantity: one number per feature and sample.",
    "sample_id": "Identifies each sample (one row per sample in a samples-in-rows table).",
    "sample_metadata": "Describes a sample (subject, time point, batch, group, covariate...).",
    "ignore": "Not used. Kept in the log, left out of the outputs.",
    "unresolved": "Not decided yet. Must be resolved before finishing.",
    "primary": "The main measurement matrix of its assay (at most one per assay).",
    "auxiliary": "Kept alongside, not used as the main matrix (e.g. iBAQ next to LFQ).",
    "excluded": "Not used and not exported.",
    "subject_id": "The individual a sample came from; repeats across repeated measures.",
    "timepoint": "When the sample was taken (visit, day, time).",
    "batch": "Processing / acquisition batch or plate.",
    "run_order": "Injection or acquisition order (drift).",
    "technical_replicate": "Marks technical replicates of the same sample.",
    "sample_type": "Says whether a sample is a study sample, QC, blank, pool or calibrator.",
    "group": ("Categorical variable that plausibly defines comparison groups. A candidate only: "
              "PRISM never decides which variable is the research outcome."),
    "covariate": "Any other sample characteristic (age, sex, CD4 count, iron...).",
    "other": "Sample information that fits none of the above.",
    "samples_in_columns": "Each row is one feature; each sample has its own column.",
    "samples_in_rows": "Each row is one sample; each feature has its own column.",
    "long": "Each row is one (feature, sample) pair with a single value column.",
    "marks_rows_as_suspect": "The column flags rows as decoy, contaminant or otherwise suspect.",
    "is_study_sample": "False for QC, blank, pool, calibrator and similar non-study injections.",
}

HISTORY_QUESTIONS = [
    ("normalized", "Were the values normalized before upload?"),
    ("log_transformed", "Were the values log-transformed before upload?"),
    ("imputed", "Were missing values imputed (filled in) before upload?"),
    ("batch_corrected", "Was a batch correction applied before upload?"),
    ("features_or_samples_removed_before_upload", "Were any features or samples removed before upload?"),
]


def vocabulary_payload():
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "vocabulary": VOCABULARY,
        "definitions": DEFINITIONS,
        "history_questions": [{"id": k, "question": q} for k, q in HISTORY_QUESTIONS],
        "scope_description": SCOPE_DESCRIPTION,
    }
