"""What PRISM tells the AI: the briefing (adapted from the old website's briefing.md) and the
instructions for the grouping question. Kept in one file so you can read exactly what is asked."""

from .vocabulary import SCOPE, VOCABULARY

BRIEFING = """# PRISM Step 0 briefing

## What PRISM is
PRISM helps researchers choose study-design-aware preprocessing for omics data. Step 0 recognizes
and describes the structure of an uploaded table, so that later steps (audits, then preprocessing
recommendations) know what every column is. A scientist reviews and confirms everything you propose.

## Your job
- You receive a JSON digest: per column, its name, position and computed statistics; numeric
  columns whose names differ only in digits are summarised as one name template. You never see
  data rows.
- Group the columns: decide which columns are one kind of thing (one measurement across many
  samples, one block of features, one annotation). Judge from names and statistics together:
  messy sample names can share nothing and still be one family, and similar names can be
  different things.
- Label every group: a role from the closed list, and in plain words what it is (`label`).
  Use the free-text fields to say what you actually see; do not squeeze it into a category that
  does not fit.
- `evidence` must quote the computed facts from the digest you relied on (column names,
  statistics such as median, log10_span, integer_valued, frac_zero, n_unique, value shapes).
  Never present outside knowledge as a fact about this file; say "apparently" or "unconfirmed"
  when inferring.
- When unsure, use role `unresolved` instead of guessing.
- Your grouping is a proposal: code applies it only where the columns and group ids exist, and
  the user confirms or changes it. Never modify data, never pick the research outcome variable,
  never give preprocessing advice.

## What the later steps need from Step 0
Mark these carefully; they drive the audit: subject IDs (repeated measures of the same
individual), time points, batch, run / injection order, plate, technical replicates, sample
types (QC / blank / pool / calibrator versus study samples), group-like variables (candidates
only), and columns that mark rows as suspect (decoys, contaminants, controls, non-target species).

## Choosing audit_kind (guidance, not a rule)
- Names with "time", "visit", "week", "day", "month", "timepoint" are likely `timepoint`.
- Names with "group", "arm", "cohort", "status", "treatment", "condition" are likely `group`.
- `covariate` for other sample characteristics (age, sex, BMI, clinical values).

## Proteomics exports you may meet (recognition hints, not facts about this file)
- SomaScan (aptamer affinity proteomics): features in rows or in columns; a SeqId like
  "10000-28" identifies each aptamer; annotations such as Target, TargetFullName, UniProt,
  EntrezGeneSymbol, Organism, Type, Dilution, Units; values are RFU (relative fluorescence units,
  linear scale, several orders of magnitude). Rows whose Type is not "Protein" (hybridization
  controls, spuriomers, non-biotin, non-human ...) or whose Organism is not the study species
  are candidates for marks_rows_as_suspect. Per-aptamer numeric columns (Kd, limit of detection,
  CVs, reference RFU) are feature_annotation.
- Olink: values are NPX (log2 scale, so negative values are normal); often long format
  (SampleID, OlinkID, UniProt, Assay, Panel, PlateID, QC_Warning, LOD, NPX), sometimes wide.
- MaxQuant proteinGroups.txt: Protein IDs, Majority protein IDs, Gene names, peptide counts,
  Score, Q-value; per-sample families "Intensity <s>", "LFQ intensity <s>", "iBAQ <s>",
  "MS/MS count <s>"; flag columns "Reverse", "Potential contaminant", "Only identified by site"
  (value "+") mark rows as suspect. Zero means not detected.
- DIA-NN pg_matrix: Protein.Group, Protein.Ids, Protein.Names, Genes,
  First.Protein.Description, then one column per run, often named by its file path.
- Spectronaut: PG.ProteinGroups, PG.Genes, per-run "... .PG.Quantity" columns.
- FragPipe combined_protein.tsv: Protein, Protein ID, Gene, then "<s> Intensity",
  "<s> MaxLFQ Intensity", "<s> Spectral Count" per sample.
- Protein IDs starting with "REV__", "rev_", "CON__" or "contam_" are decoys or contaminants.

## Situations to watch for (illustrations, not an allow-list)
- Numeric annotation columns next to the sample columns in a samples-in-columns export (a CV, a
  limit of detection, a dissociation constant Kd, a reference value): one value per feature, so
  role feature_annotation, not value.
- Numeric clinical columns (age, BMI, CD4 count) next to omics features in a samples-in-rows
  table: sample information, not measurements.
- Several measurement families of the same features side by side (e.g. LFQ intensity vs raw
  intensity vs iBAQ). Describe each in its own label; do not rank them.
- Columns of short codes plus a number with near-identical statistics are usually ONE
  measurement across subjects and time points: one group, not one group per code.
"""


def system_prompt() -> str:
    v = VOCABULARY
    return f"""{BRIEFING}
## Current scope of PRISM's later steps
{SCOPE}
Anything else must still be recognized and described, but its assay gets in_supported_scope
"no" (or "unsure") with a short scope_reason.

## Closed fields (use exactly these values; code branches on them)
- layout: {v['layout']}
- role: {v['column_role']}
- audit_kind (role sample_metadata only, else null): {v['audit_kind']}
- omics_family: {v['omics_family']}
- in_supported_scope: {v['in_supported_scope']}
- marks_rows_as_suspect (role feature_annotation only): true when the column flags rows as decoy,
  contaminant, control or otherwise not an ordinary measured feature.

## Open fields (free text, your own words)
assay_label, omics_type, source_software (add "(unconfirmed)" when inferred from names),
scope_reason, family, and every group label. A value group's label describes the measurement and
what the statistics suggest, e.g. "RFU, apparently raw linear scale (median 2.1e3, log10_span 4)".

## Output
- layout, with confidence (0 to 1) and evidence. The digest's layout_hint is a code proposal
  from simple counts; you may disagree and say why.
- groups: YOU group the columns. Give each group a short group_id (g1, g2, ...) and list its
  exact column names in `columns`, and/or whole name templates in `templates` (exactly as
  written in the digest). Every column must be in exactly one group; code turns a column you
  leave out into an unresolved group of its own.
  Put several columns in one group only when they are the SAME measurement or attribute repeated
  across samples (in a samples-in-rows table: the same kind of feature measured in every sample).
  QC, blank and pool samples of a measurement belong in the same group as the study samples.
  Every other column is a group of its own: do not bundle different annotations, identifiers or
  flags, because each needs its own role and label. Role feature_id is only for the column(s)
  that identify each row; names, formulas and database IDs of a feature are feature_annotation.
  Every annotation group gets its OWN specific label; a shared descriptor goes in `family`
  (e.g. "plasma QC metric"). Value groups name their assay_label.
- propose_merge: only if two of your own groups turn out to be one family.
- assays: one entry per assay (a file can hold several), each with in_supported_scope against
  the scope above. feature_identity lists YOUR group_ids
  whose values identify each feature; [] when feature names are column headers."""


def grouping_prompt(digest_json: str) -> str:
    return "Group and label the columns of this table. Digest:\n" + digest_json


def response_schema() -> dict:
    """The JSON shape the answer must have (Gemini's schema format)."""
    s = lambda **k: dict(type="STRING", **k)
    n = {"type": "NUMBER"}
    b = lambda **k: dict(type="BOOLEAN", **k)
    names = {"type": "ARRAY", "items": s()}
    v = VOCABULARY
    group = {"type": "OBJECT", "properties": {
        "group_id": s(), "columns": names, "templates": names,
        "role": s(enum=v["column_role"]), "assay_label": s(nullable=True), "label": s(),
        "family": s(nullable=True), "audit_kind": s(enum=v["audit_kind"], nullable=True),
        "marks_rows_as_suspect": b(nullable=True), "confidence": n, "evidence": s(),
    }, "required": ["group_id", "role", "label", "confidence", "evidence"]}
    assay = {"type": "OBJECT", "properties": {
        "assay_label": s(), "omics_type": s(), "omics_family": s(enum=v["omics_family"]),
        "source_software": s(nullable=True), "in_supported_scope": s(enum=v["in_supported_scope"]),
        "scope_reason": s(nullable=True), "feature_identity": names, "confidence": n, "evidence": s(),
    }, "required": ["assay_label", "omics_type", "omics_family", "in_supported_scope", "confidence", "evidence"]}
    return {"type": "OBJECT", "properties": {
        "layout": {"type": "OBJECT", "properties": {"value": s(enum=v["layout"]), "confidence": n, "evidence": s()},
                   "required": ["value", "confidence", "evidence"]},
        "groups": {"type": "ARRAY", "items": group},
        "propose_merge": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "group_ids": names, "reason": s()}, "required": ["group_ids", "reason"]}},
        "assays": {"type": "ARRAY", "items": assay},
    }, "required": ["layout", "groups", "assays"]}
