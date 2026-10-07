"""AI layer: the AI groups the columns and labels the groups, from facts.

What is sent: file-level layout hints and, per column, its name, position and
computed statistics, plus the name parts it shares with other columns
(shared_prefix / shared_suffix: a fact, not a grouping). Never raw rows.
Example values only for low-cardinality text columns (<= 20 distinct values),
and only when AI_SEND_EXAMPLE_VALUES is on (default).

Grouping is judgment, so it is the AI's proposal (v2.2): which columns are one
family, what each group is. Wide files are sent in chunks of
config.GROUPING_CHUNK_SIZE columns (a prompt size limit) and a consolidation
call lets the AI join families split across chunks. Code only applies
proposals whose ids and column names exist (grouping.py) and puts every column
nobody mentioned into its own 'unresolved' group.

What comes back is parsed strictly (Pydantic) and checked structurally
(validation.py): values outside the closed sets and contradicted claims
become 'unresolved'.
"""

from __future__ import annotations

import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ValidationError

from . import grouping, llm_providers as llm
from . import config
from .config import SCOPE_DESCRIPTION
from .profiling import chunk_columns, chunk_units, common_pattern, name_templates, near_duplicate_pairs
from .schema import DEFINITIONS, HISTORY_QUESTIONS, PROMPT_VERSION, UNRESOLVED, VOCABULARY
from .validation import timepoint_detail, validate_feature_identity, validate_group

CACHE_DIR = Path(os.environ.get("PRISM_CACHE_DIR", Path(__file__).parent / "cache"))


def send_examples():
    return os.environ.get("AI_SEND_EXAMPLE_VALUES", "true").strip().lower() not in ("0", "false", "no", "off")


# ---------------------------------------------------------------- digest

def column_digest(cols, i, affix, examples):
    d = dict(cols.digests[i])
    if not examples or (d.get("unique_ratio") or 0) >= 0.5:
        # identifier-like columns (e.g. patient codes) never send example values
        d.pop("values", None)
    if not examples:
        d.pop("value_shapes", None)
    d["shared_prefix"] = affix.get("shared_prefix")
    d["shared_suffix"] = affix.get("shared_suffix")
    return d


def template_digest(cols, tpl):
    """One entry for a name template (columns whose names differ only in digits)."""
    names = [cols.labels[i] for i in tpl["indices"]]
    x = {"template": tpl["template"], "n_columns": tpl["n_columns"], "first_position": tpl["indices"][0] + 1,
         "types": tpl["types"], "columns": names if len(names) <= 12 else names[:3] + [f"... ({len(names) - 6} more)"] + names[-3:]}
    if tpl.get("aggregate"):
        x["aggregate"] = tpl["aggregate"]
    if set(tpl["types"]) != {"numeric"}:
        first = dict(cols.digests[tpl["indices"][0]])
        first.pop("values", None)
        x["first_column"] = first
    return x


def build_digest(filename, cols, affixes, hints, indices, fixed=None, signature_hint=None, chunk=None, templates=None):
    """Templates first (v2.4 §15): columns whose names differ only in digits are one entry with an
    aggregate profile; every other column is listed with its own digest."""
    examples = send_examples()
    if templates is None:
        templates = name_templates(cols, indices)
    multi = [t for t in templates if t["n_columns"] >= 2]
    singles = sorted(t["indices"][0] for t in templates if t["n_columns"] == 1)
    d = {
        "file": {"name_extension": Path(filename).suffix.lower(), "layout_hints": hints,
                 "signature_hint": signature_hint, "n_columns_in_file": len(cols.header),
                 "n_columns_in_this_call": len(indices)},
        "already_confirmed": fixed or {},
        "templates": [template_digest(cols, t) for t in multi],
        "columns": [column_digest(cols, i, affixes[i], examples) for i in singles],
        "settings": {"example_values_sent": examples, "raw_rows_sent": False},
    }
    cand = sorted(i for t in templates for i in t["indices"] if t["n_columns"] < 5 or cols.digests[i]["type"] != "numeric")
    pairs, skipped = near_duplicate_pairs(cols, cand)
    d["file"]["near_duplicate_columns"] = pairs if not skipped else {"skipped": skipped}
    num = [cols.labels[i] for i in indices if cols.digests[i]["type"] == "numeric"]
    if 2 <= len(num) <= 20000:
        from .design import derivation_candidates
        d["file"]["sample_name_facts"] = {
            "from": "numeric column names (sample names when samples are in columns)",
            "derivation_candidates": derivation_candidates(num)}
    if multi:
        d["templates_note"] = ("Each template stands for several columns whose names differ only in their digits "
                               "(digits shown as '#'). Put a whole template in a group with `templates`; name single "
                               "member columns in `columns` only to split a deviant out.")
    if chunk:
        d["chunk"] = {"index": chunk[0], "of": chunk[1], "n_columns": len(indices),
                      "note": ("Only these columns are in this call. Your groups and labels here are drafts: a final "
                               "call sees every chunk's proposal and decides the final groups, labels and assays.")}
    return d


def digest_hash(digest):
    return hashlib.sha256(json.dumps(digest, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- prompt

BRIEFING = (Path(__file__).parent / "briefing.md").read_text(encoding="utf-8")


def system_prompt():
    closed = VOCABULARY
    return f"""{BRIEFING}

## Current scope of PRISM's later steps
{SCOPE_DESCRIPTION}
Anything else must still be recognized and described, but flagged in_supported_scope "no" (or
"unsure") with a short scope_reason.

## Closed fields (use exactly these values; code branches on them)
- layout: {closed['layout']}
- role: {closed['column_role']}
- audit_kind (role sample_metadata only, else null): {closed['audit_kind']} ('other' is the escape hatch)
- in_supported_scope: {closed['in_supported_scope']}
- marks_rows_as_suspect (feature_annotation only): true when the column flags rows as decoy /
  contaminant / otherwise suspect.
- is_study_sample: false for QC, blank, pool, calibrator and similar injections.

## Open fields (free text, your own words)
assay_label, omics_type, source_software (add "(unconfirmed)" when inferred from names),
scope_reason, and every group label. A value block's label should describe the measurement and
what the statistics suggest, e.g. "LFQ intensity, apparently raw linear scale (median 2.1e7,
18% zeros)". A time point's label should say what kind of time it is, e.g. "visit label T1/T2
(ordinal)".

## Output
- layout, with confidence and evidence.
- groups: YOU group the columns. A group is a set of columns that are one kind of thing: one
  measurement taken for many samples (e.g. "LFQ intensity S01" ... "LFQ intensity S24"), one
  block of features measured in every sample, or a single annotation / sample column (a group
  of one). Give each group your own short group_id (g1, g2, ...) and list its exact column
  names in `columns`, or whole name templates in `templates` (listed first in the digest: columns
  whose names differ only in digits, with an aggregate profile). Every column must be in exactly
  one group; code turns a column you leave out into an unresolved group of its own.
  shared_prefix / shared_suffix only say which literal name parts a column shares with others;
  they are hints, not groups. Names without shared text (Pt003_visit1, 004-w1, Glucose,
  Lactate) can still be one family when the statistics agree; columns with a shared prefix can
  still be different things. Do not put different kinds of thing in one group (e.g. a clinical
  covariate among metabolites, a total over all samples next to the per-sample columns).
  Put several columns in one group only when they are the SAME measurement or attribute
  repeated across samples (in samples-in-rows files: the same kind of feature measured in every
  sample). QC, blank, pool and calibrator samples of that measurement belong in the same group
  as the study samples; describe them in `samples`, not with separate groups (e.g. "QC_01 Peak
  area", "blank_01 Peak area" and "S01 Peak area" are ONE group: peak area for every sample). Every other column
  is a group of its own: do not bundle different annotations, identifiers or flag columns
  together, because each needs its own role, label and (for flags) its own flagged value.
  Role feature_id is only for the column(s) that identify each row; names, formulas and database
  IDs of a feature are feature_annotation. Value groups name their assay_label.
- propose_merge: only if two of your own groups turn out to be one family.
- suggest_split (with suggest_split_role / _audit_kind / _label): columns inside a group that
  are a different kind of thing; used when you are asked to reconsider an existing group.
- assays: one entry per assay (a file can hold several, e.g. proteins and metabolites side by
  side). feature_identity lists YOUR group_ids whose values identify each feature (several =
  composite key such as m/z + retention time); use [] when feature names are column headers.
- samples: sample names or glob patterns (e.g. "QC_*") with a label and is_study_sample.
- clarifying_questions for anything you cannot resolve from the digest: a question with type
  (single / multi / confirm), text, applies_to {{columns}}, and clickable options; each option may
  carry the patches it would apply (same format and ops as in the chat). Ask instead of guessing.
- omics_family per assay: one value of the closed list {closed['omics_family']}; omics_type stays your
  own words.
- Every annotation column gets its OWN specific label (what this column is, from its digest: value
  shapes, ranges, repeated values); a shared descriptor goes in `family` (e.g. "plasma QC metric").
- For low-cardinality text annotation columns, consider whether a value marks rows that are not
  ordinary measured features (non-target species, controls, spike-ins, decoys, contaminants). If
  plausible, ask a multi question "Which values mark rows to flag?" with one option per value
  (with its count) carrying set_flag_values. Work only from the names and values in the digest.
- near_duplicate_columns (a fact) lists pairs of columns with (nearly) identical values; you may
  ask whether to keep both. Never drop one yourself.
- processing_hints (optional): a short hint per processing-history question when the statistics
  suggest something (e.g. "every column's median is close to 1: possibly median-scaled"). A hint
  raises the question for the user; it never answers it. Platform exports may already include
  normalization.
- design (optional): where the subject (the individual a sample came from) and the time point come
  from: source metadata_column (a sample information column), derived_from_sample_names (with a
  derivation {{delimiter, occurrence first|last, left / right: subject | time}}; see
  file.sample_name_facts) or none. Never state the time unit: it is asked.
- Respect everything under already_confirmed."""


def design_schema():
    s = lambda **k: dict(type="STRING", **k)
    src = {"type": "OBJECT", "properties": {"source": s(enum=["metadata_column", "derived_from_sample_names", "none"]),
                                            "column": s(nullable=True), "file": s(enum=["main", "metadata"], nullable=True)}}
    return {"type": "OBJECT", "nullable": True, "properties": {
        "subject": src, "time": src,
        "derivation": {"type": "OBJECT", "nullable": True, "properties": {
            "delimiter": s(), "occurrence": s(enum=["first", "last"]), "left": s(nullable=True), "right": s(nullable=True)}},
        "confidence": {"type": "NUMBER"}, "evidence": s()}}


def response_schema():
    s = lambda **k: dict(type="STRING", **k)
    n = lambda: {"type": "NUMBER"}
    b = lambda **k: dict(type="BOOLEAN", **k)
    fact = {"type": "OBJECT", "properties": {"value": s(enum=VOCABULARY["layout"]), "confidence": n(),
                                             "evidence": s()}, "required": ["value", "confidence", "evidence"]}
    assay = {"type": "OBJECT", "properties": {
        "assay_label": s(), "omics_type": s(), "omics_family": s(enum=VOCABULARY["omics_family"]), "source_software": s(),
        "in_supported_scope": s(enum=VOCABULARY["in_supported_scope"]), "scope_reason": s(),
        "feature_identity": {"type": "OBJECT", "properties": {
            "group_ids": {"type": "ARRAY", "items": s()}, "composite": b()}, "required": ["group_ids"]},
        "confidence": n(), "evidence": s()},
        "required": ["assay_label", "omics_type", "in_supported_scope", "confidence", "evidence"]}
    group = {"type": "OBJECT", "properties": {
        "group_id": s(),
        "columns": {"type": "ARRAY", "items": s()},
        "templates": {"type": "ARRAY", "items": s()},
        "role": s(enum=VOCABULARY["column_role"]),
        "assay_label": s(nullable=True),
        "label": s(),
        "family": s(nullable=True),
        "audit_kind": s(enum=VOCABULARY["audit_kind"], nullable=True),
        "marks_rows_as_suspect": b(nullable=True),
        "confidence": n(),
        "evidence": s(),
        "suggest_split": {"type": "ARRAY", "items": s(), "nullable": True},
        "suggest_split_role": s(enum=VOCABULARY["column_role"], nullable=True),
        "suggest_split_audit_kind": s(enum=VOCABULARY["audit_kind"], nullable=True),
        "suggest_split_label": s(nullable=True),
    }, "required": ["group_id", "role", "label", "confidence", "evidence"]}
    return {"type": "OBJECT", "properties": {
        "layout": fact,
        "assays": {"type": "ARRAY", "items": assay},
        "groups": {"type": "ARRAY", "items": group},
        "samples": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "pattern_or_sample": s(), "label": s(), "is_study_sample": b(),
            "confidence": n(), "evidence": s()}, "required": ["pattern_or_sample", "label", "is_study_sample"]}},
        "clarifying_questions": questions_schema(),
        "propose_merge": merges_schema(),
        "design": design_schema(),
        "processing_hints": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "question": s(enum=[q for q, _ in HISTORY_QUESTIONS]), "hint": s()}, "required": ["question", "hint"]}},
    }, "required": ["layout", "assays", "groups"]}


def merges_schema():
    return {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "group_ids": {"type": "ARRAY", "items": {"type": "STRING"}}, "reason": {"type": "STRING"}},
        "required": ["group_ids", "reason"]}}


CONSOLIDATION_PROMPT = """

## This call: the final grouping of a file sent in chunks
The file was too wide for one call, so it was sent in chunks. You get every chunk's proposal: its
groups (id, chunk, size, role, draft label, templates, first / last columns, the name part they
share, one line of statistics) and the assays each chunk named. Return the FINAL answer for the
whole file:
- groups: each final group lists the chunk group ids it is made of in `members` (several when one
  family was split across chunks), with its role, assay_label, label (and family, audit_kind,
  marks_rows_as_suspect where they apply), confidence and evidence. Labels from chunks are
  drafts: write the final labels yourself and never mention chunks.
- assays: the assays of the whole file. An assay is a separate measurement (its own platform or
  run, usually its own feature set and scale). Classes of features inside one export (pathway,
  protein family, lipid class, metabolite super-pathway) are annotations of features, not assays.
  When unsure whether blocks are one measurement or several, ask in clarifying_questions.
- Every chunk group should be in exactly one final group. Leave out only what you cannot place:
  it becomes a question for the user."""


METADATA_PROMPT = """

## This call: a sample metadata file
You get the digest of a sample information file (usually one row per sample): per column its
computed statistics, and join facts: how many of the data file's sample names its values contain
(exact, and after normalising case, spaces, separators and leading zeros). Return:
- join_key: the column whose values name the data's samples (use the join facts), with
  confidence and evidence. The user confirms it.
- columns: EVERY column of the file: role (sample_id for the join key only, sample_metadata, or
  ignore), audit_kind from the closed list, a specific label in your own words (what this column
  is; two columns must not share a vague label), confidence, and evidence quoting the digest.
  Two columns that mean the same thing (e.g. time_point and TimePoint) get the same audit_kind.
- design: where the subject and the time come from (a metadata column, a rule over the sample
  names using data.derivation_candidates, or none). Never state the time unit: it is asked.
- clarifying_questions (with options) for anything you cannot decide from the digest."""


RELABEL_PROMPT = """

## This call: column-specific labels
Several annotation columns were all given the same label. For EACH column listed, write its own
specific label (what this column is, from its digest: value shapes, ranges, repeated values,
name). Put a descriptor they share in `family`. Never reuse one label for several columns."""


PATCH_OPS = ["set_keep", "set_role", "set_audit_kind", "set_label", "set_block_keep", "merge_groups", "split_group",
             "merge_assays", "derive_feature_annotation", "set_design", "set_sample_label", "set_join_key",
             "set_flag_values"]

CHAT_PROMPT = """

## This call: the user talks to you about the schema (chat)
You get the user's message, the current step, the columns they selected, a compact summary of
the draft schema (groups with role / label / columns, every annotation and sample-information
column, assays, design, open questions, excluded columns), the computed digest of the columns
the message refers to, and the last chat turns. Answer in `reply` (short, concrete, plain
words). You act on the schema ONLY through `patches`, from this closed list of ops:
- set_keep {keep}: keep false = leave the targeted columns out of the outputs (nothing is deleted;
  they are listed with a reason). set_block_keep {keep}: the same for whole value blocks.
- set_role {role}; set_audit_kind {audit_kind, detail?} (sample information only);
  set_label {label, family?} (family: an optional shared descriptor such as "plasma QC metric").
- merge_groups: the targeted columns' groups are one group. split_group: take the targeted columns
  out of their group.
- merge_assays {assay_labels, label?}: these assays are one measurement.
- derive_feature_annotation {source: "column_headers" | "feature_id_column", column?, rule:
  {delimiter, occurrence: "first" | "last", parts: [{name, label}, {name, label}]}}: new annotation
  columns from parts of the feature names (names are never changed).
- set_sample_label {label?, is_study_sample?} with target.selector.samples (names or patterns
  such as "QC_*").
- set_join_key {metadata_column}: the metadata column that names the samples.
- set_flag_values {flagged_values}: which values of one annotation column mark rows as flagged.
Targets: {"selector": {...}, "except_columns": [...]}. Selector keys (combined with AND): columns
(exact names), group_id / group_ids, role, audit_kind, file ("main" | "metadata"),
name_contains / name_starts_with / name_ends_with (plain text, no regex), samples. Use only names
and ids from the summary; an invented name is rejected. Give each patch a patch_id, a one-line
reason, and consequences: what later steps could lose (e.g. excluding a plate, batch, run-order,
QC-metric, sample-type or suspect-row flag column).
Rules:
- Say "I've prepared a change" — never claim it is done; the user applies it.
- If the request is ambiguous, ask with a question that has options (each option carries the
  patches it would apply) instead of guessing.
- The user is the authority: do what they confirm; refuse only what an invariant forbids (the
  feature ID or sample ID column cannot be excluded; at least one value block stays kept; no op
  changes values, renames source columns, touches processing history or finalizes).
- Do not choose the research outcome variable, do not give preprocessing advice and do not answer
  the processing-history questions: say that comes in a later step."""


def _patch_schema():
    s = lambda **k: dict(type="STRING", **k)
    b = lambda **k: dict(type="BOOLEAN", **k)
    arr = lambda: {"type": "ARRAY", "items": {"type": "STRING"}}
    selector = {"type": "OBJECT", "properties": {
        "columns": arr(), "group_id": s(nullable=True), "group_ids": arr(),
        "role": s(enum=VOCABULARY["column_role"], nullable=True), "audit_kind": s(enum=VOCABULARY["audit_kind"], nullable=True),
        "file": s(enum=["main", "metadata"], nullable=True), "name_contains": s(nullable=True),
        "name_starts_with": s(nullable=True), "name_ends_with": s(nullable=True), "samples": arr()}}
    part = {"type": "OBJECT", "properties": {"name": s(), "label": s(nullable=True)}}
    rule = {"type": "OBJECT", "properties": {"delimiter": s(), "occurrence": s(enum=["first", "last"]),
                                             "parts": {"type": "ARRAY", "items": part},
                                             "left": s(nullable=True), "right": s(nullable=True)}}
    source = {"type": "OBJECT", "properties": {
        "source": s(enum=["metadata_column", "derived_from_sample_names", "none"]), "column": s(nullable=True),
        "file": s(enum=["main", "metadata"], nullable=True)}}
    args = {"type": "OBJECT", "properties": {
        "keep": b(nullable=True), "role": s(enum=VOCABULARY["column_role"], nullable=True),
        "audit_kind": s(enum=VOCABULARY["audit_kind"], nullable=True), "detail": s(nullable=True),
        "label": s(nullable=True), "family": s(nullable=True), "flagged_values": arr(), "assay_labels": arr(),
        "metadata_column": s(nullable=True), "is_study_sample": b(nullable=True),
        "source": s(enum=["column_headers", "feature_id_column"], nullable=True), "column": s(nullable=True),
        "rule": rule, "subject": source, "time": source, "derivation": rule}}
    return {"type": "OBJECT", "properties": {
        "patch_id": s(), "op": s(enum=PATCH_OPS),
        "target": {"type": "OBJECT", "properties": {"selector": selector, "except_columns": arr()}},
        "args": args, "reason": s(), "consequences": arr()}, "required": ["patch_id", "op", "reason"]}


def questions_schema():
    s = lambda **k: dict(type="STRING", **k)
    return {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "type": s(enum=["single", "multi", "confirm"]), "text": s(),
        "applies_to": {"type": "OBJECT", "properties": {"columns": {"type": "ARRAY", "items": s()},
                                                         "group_ids": {"type": "ARRAY", "items": s()}}},
        "step": s(nullable=True),
        "options": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "label": s(), "patches": {"type": "ARRAY", "items": _patch_schema()}}, "required": ["label"]}},
        "allow_free_text": {"type": "BOOLEAN", "nullable": True}}, "required": ["type", "text", "options"]}}


def chat_schema():
    return {"type": "OBJECT", "properties": {
        "reply": {"type": "STRING"}, "patches": {"type": "ARRAY", "items": _patch_schema()},
        "questions": questions_schema()}, "required": ["reply"]}


AUDIT_OPERATOR_PROMPT = """
You are the operator of the PRISM Tier 1 audit. The audit is deterministic code; you never compute
numbers yourself and you never write code that runs. You receive the user's message and, as context,
each dataset's schema and the finding JSON of the audit run being viewed (never raw data rows).

Answer in "reply", and call tools in "tool_calls" (each: tool + args_json, a JSON object as a string):
  run_audit        {"factors": ["A4", ...] or [] for all, "datasets": ["D1"] or [], "params": {...}}
  rerun            {}                       re-run the run being viewed with its own parameters
  show_finding     {"finding": "A4__D1_A1.json"}
  set_override     {"kind": ..., "sample" | "column" | "key": ..., "role" | "value": ..., "reason": ...}
                   (a proposal: the user applies it)
  plot             {"finding": "A7__D1_A1.json", "color_by": "<variable>", "pcs": [1, 2]}
  request_addition {"title", "inputs", "outputs", "algorithm", "tests", "citations"}
                   (anything the audit registry cannot do: a written request, nothing runs)
Rules:
- Parameters come from the user's message or the defaults. Never change a parameter the user did not
  name with its value.
- Explanations use only numbers that appear in the findings. Use the indicator sentences; you may
  rephrase them, never invent new ones. Say "consistent with" or "plausible indicator", never
  "diagnosed" or "confirmed".
- You cannot answer processing-history questions: those are the user's.
"""


def audit_chat_schema():
    return {"type": "OBJECT", "properties": {
        "reply": {"type": "STRING"},
        "tool_calls": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "tool": {"type": "STRING", "enum": ["run_audit", "rerun", "show_finding", "set_override", "plot",
                                                 "request_addition"]},
            "args_json": {"type": "STRING"}}, "required": ["tool", "args_json"]}}},
        "required": ["reply"]}


def metadata_schema():
    s = lambda **k: dict(type="STRING", **k)
    col = {"type": "OBJECT", "properties": {
        "column": s(), "role": s(enum=["sample_id", "sample_metadata", "ignore"]),
        "audit_kind": s(enum=VOCABULARY["audit_kind"], nullable=True), "label": s(), "family": s(nullable=True),
        "detail": s(nullable=True), "confidence": {"type": "NUMBER"}, "evidence": s()},
        "required": ["column", "role", "label", "confidence", "evidence"]}
    return {"type": "OBJECT", "properties": {
        "join_key": {"type": "OBJECT", "nullable": True, "properties": {
            "column": s(), "confidence": {"type": "NUMBER"}, "evidence": s()}},
        "columns": {"type": "ARRAY", "items": col}, "design": design_schema(),
        "clarifying_questions": questions_schema()}, "required": ["columns"]}


def relabel_schema():
    s = lambda **k: dict(type="STRING", **k)
    return {"type": "OBJECT", "properties": {"labels": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "column": s(), "label": s(), "family": s(nullable=True)}, "required": ["column", "label"]}}},
        "required": ["labels"]}


def consolidation_schema():
    s = lambda **k: dict(type="STRING", **k)
    final = {"type": "OBJECT", "properties": {
        "group_id": s(), "members": {"type": "ARRAY", "items": s()}, "role": s(enum=VOCABULARY["column_role"]),
        "assay_label": s(nullable=True), "label": s(), "family": s(nullable=True),
        "audit_kind": s(enum=VOCABULARY["audit_kind"], nullable=True),
        "marks_rows_as_suspect": {"type": "BOOLEAN", "nullable": True}, "confidence": {"type": "NUMBER"},
        "evidence": s()}, "required": ["group_id", "members", "role", "label", "confidence", "evidence"]}
    return {"type": "OBJECT", "properties": {
        "groups": {"type": "ARRAY", "items": final},
        "assays": {"type": "ARRAY", "items": response_schema()["properties"]["assays"]["items"]},
        "clarifying_questions": questions_schema(), "comment": s(nullable=True)},
        "required": ["groups", "assays"]}


# ---------------------------------------------------------------- parsing (Pydantic)

class Fact(BaseModel):
    value: str
    confidence: float = 0.0
    evidence: str = ""


class FeatureIdentity(BaseModel):
    group_ids: List[str] = []
    composite: Optional[bool] = False


class Assay(BaseModel):
    assay_label: str
    omics_type: str = "unknown"
    omics_family: Optional[str] = "unknown"
    source_software: Optional[str] = "unknown"
    in_supported_scope: str = "unsure"
    scope_reason: Optional[str] = ""
    feature_identity: Optional[FeatureIdentity] = None
    confidence: float = 0.0
    evidence: str = ""


class GroupLabel(BaseModel):
    group_id: str
    columns: List[str] = []
    templates: List[str] = []
    family: Optional[str] = None
    role: str
    assay_label: Optional[str] = None
    label: Optional[str] = ""
    audit_kind: Optional[str] = None
    marks_rows_as_suspect: Optional[bool] = None
    confidence: float = 0.0
    evidence: str = ""
    suggest_split: Optional[List[str]] = None
    suggest_split_role: Optional[str] = None
    suggest_split_audit_kind: Optional[str] = None
    suggest_split_label: Optional[str] = None


class SampleRule(BaseModel):
    pattern_or_sample: str
    label: str = ""
    is_study_sample: bool = True
    confidence: float = 0.0
    evidence: str = ""


class Merge(BaseModel):
    group_ids: List[str] = []
    reason: str = ""


class FinalGroup(BaseModel):
    group_id: str
    members: List[str] = []
    role: str
    assay_label: Optional[str] = None
    label: Optional[str] = ""
    family: Optional[str] = None
    audit_kind: Optional[str] = None
    marks_rows_as_suspect: Optional[bool] = None
    confidence: float = 0.0
    evidence: str = ""


class Patch(BaseModel):
    patch_id: str = ""
    op: str
    target: dict = {}
    args: dict = {}
    reason: str = ""
    consequences: List[str] = []


class QOption(BaseModel):
    label: str
    patches: List[Patch] = []


class AIQuestion(BaseModel):
    type: str = "single"
    text: str = ""
    question: Optional[str] = None      # older shape: {group_id, question}
    group_id: Optional[str] = None
    applies_to: dict = {}
    step: Optional[str] = None
    options: List[QOption] = []
    allow_free_text: Optional[bool] = True


class MetaColumn(BaseModel):
    column: str
    role: str = "sample_metadata"
    audit_kind: Optional[str] = None
    label: Optional[str] = ""
    family: Optional[str] = None
    detail: Optional[str] = None
    confidence: float = 0.0
    evidence: str = ""


class MetadataResponse(BaseModel):
    join_key: Optional[dict] = None
    columns: List[MetaColumn] = []
    design: Optional[dict] = None
    clarifying_questions: List[AIQuestion] = []


class RelabelResponse(BaseModel):
    labels: List[dict] = []


class ChatResponse(BaseModel):
    reply: str = ""
    patches: List[Patch] = []
    questions: List[AIQuestion] = []


class AuditToolCall(BaseModel):
    tool: str
    args_json: str = "{}"


class AuditChatResponse(BaseModel):
    reply: str = ""
    tool_calls: List[AuditToolCall] = []


class ConsolidationResponse(BaseModel):
    groups: List[FinalGroup] = []
    assays: List[Assay] = []
    clarifying_questions: List[AIQuestion] = []
    comment: Optional[str] = ""


class AIResponse(BaseModel):
    design: Optional[dict] = None
    processing_hints: List[dict] = []
    propose_merge: List[Merge] = []
    layout: Optional[Fact] = None
    assays: List[Assay] = []
    groups: List[GroupLabel] = []
    samples: List[SampleRule] = []
    clarifying_questions: List[AIQuestion] = []


def _clamp(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------- calling

def _cache_key(sha, model_key, digest, kind="digest"):
    return hashlib.sha256(f"{sha}|{PROMPT_VERSION}|{kind}|{model_key}|{digest_hash(digest)}".encode()).hexdigest()[:40]


def ask(digest, sha, log=None, mock_fn=None):
    return _call(digest, sha, log, mock_fn)


def _call(digest, sha, log=None, mock_fn=None, kind="digest"):
    """One validated-JSON round trip with a single retry on invalid output; cached on
    disk by file, prompt version, models and the exact payload (so the chunk contents).
    Returns (parsed | None, meta). meta has provider, model, error, raw."""
    header, system, schema, model_cls = {
        "digest": ("Digest (JSON):", system_prompt(), response_schema(), AIResponse),
        "consolidation": ("Consolidation (JSON):", BRIEFING + CONSOLIDATION_PROMPT, consolidation_schema(),
                          ConsolidationResponse),
        "chat": ("Chat (JSON):", BRIEFING + CHAT_PROMPT, chat_schema(), ChatResponse),
        "metadata": ("Metadata (JSON):", BRIEFING + METADATA_PROMPT, metadata_schema(), MetadataResponse),
        "relabel": ("Relabel (JSON):", BRIEFING + RELABEL_PROMPT, relabel_schema(), RelabelResponse),
        "audit_chat": ("AuditChat (JSON):", AUDIT_OPERATOR_PROMPT, audit_chat_schema(), AuditChatResponse),
    }[kind]
    ok, why = llm.available()
    meta = {"provider": llm.provider_name(), "model": llm.model_list()[0] if ok else None,
            "prompt_version": PROMPT_VERSION, "temperature": llm.TEMPERATURE,
            "digest_hash": digest_hash(digest), "cached": False, "error": None}
    if not ok:
        meta["error"] = why
        return None, meta
    key = _cache_key(sha, f"{meta['provider']}:{','.join(llm.model_list())}", digest, kind)
    cache_file = CACHE_DIR / f"{key}.json"
    prompt = header + "\n" + json.dumps(digest, ensure_ascii=False)
    if log:
        log("ai_request", {"kind": kind, "provider": meta["provider"], "models": llm.model_list(),
                           "prompt_version": PROMPT_VERSION, "temperature": llm.TEMPERATURE,
                           "digest_hash": meta["digest_hash"], "digest": digest})
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            parsed = model_cls.model_validate_json(cached["raw"])
            meta.update(model=cached.get("model"), cached=True, raw=cached["raw"])
            llm.say(f"using cached proposal ({cache_file.name})")
            if log:
                log("ai_response_raw", {"cached": True, "model": cached.get("model"), "raw": cached["raw"]})
            return parsed, meta
        except (ValueError, KeyError, ValidationError):
            pass
    last_err = None
    for attempt in (1, 2):
        try:
            raw, m = llm.complete_json(system, prompt, schema, mock_fn)
        except llm.LLMError as e:
            meta["error"] = str(e)
            llm.say(f"FAILED: {e}")
            if log:
                log("ai_response_raw", {"error": str(e)})
            return None, meta
        except Exception as e:
            traceback.print_exc()
            meta["error"] = f"Unexpected error: {type(e).__name__}: {e}"
            return None, meta
        meta.update(model=m["model"], finish_reason=m["finish_reason"], raw=raw)
        if log:
            log("ai_response_raw", {"attempt": attempt, "model": m["model"], "finish_reason": m["finish_reason"],
                                    "raw": raw})
        try:
            if m["finish_reason"] != "STOP":
                raise ValueError(f"finish_reason={m['finish_reason']}")
            parsed = model_cls.model_validate_json(raw)
        except (ValueError, ValidationError) as e:
            last_err = f"invalid JSON from the model ({str(e).splitlines()[0][:200]})"
            llm.say(f"attempt {attempt}: {last_err}" + (" -> retrying once" if attempt == 1 else ""))
            continue
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"model": m["model"], "raw": raw}), encoding="utf-8")
        except OSError:
            pass
        return parsed, meta
    meta["error"] = last_err
    return None, meta


# ---------------------------------------------------------------- validate & assemble

GROUP_KEYS = ("role", "assay_label", "label", "audit_kind", "marks_rows_as_suspect")


def _item(gl):
    return {"role": gl.role, "assay_label": gl.assay_label, "label": (gl.label or "").strip(),
            "family": (gl.family or "").strip() or None,
            "audit_kind": gl.audit_kind, "marks_rows_as_suspect": bool(gl.marks_rows_as_suspect),
            "confidence": _clamp(gl.confidence), "evidence": gl.evidence, "source": "ai"}


def _split_item(gl, evidence):
    return {"role": gl.suggest_split_role or UNRESOLVED, "assay_label": gl.assay_label, "audit_kind": gl.suggest_split_audit_kind,
            "label": gl.suggest_split_label or "", "marks_rows_as_suspect": False, "confidence": 0.7, "source": "ai",
            "evidence": f"Split out of the group on the AI's suggestion: {evidence}"}


def read_chunk(resp, indices, label_to_idx, ns, out, tmap=None):
    """Validate one chunk's answer into out (layout, assays, pgroups, ...). Templates a group
    names are expanded to their member columns in this call; an unknown template is rejected."""
    rej = out["rejected"]
    if resp.design and not out.get("design"):
        out["design"] = resp.design
    for h in resp.processing_hints or []:
        if h.get("question") in dict(HISTORY_QUESTIONS) and h.get("hint"):
            out.setdefault("processing_hints", {}).setdefault(h["question"], str(h["hint"])[:300])
    tmap, labels = tmap or {}, {i: lab for lab, i in label_to_idx.items()}
    for gl in resp.groups:
        for tp in gl.templates or []:
            if tp not in tmap:
                rej.append({"group_id": gl.group_id, "template": tp,
                            "reason": "This name template does not exist in this call; the claim was rejected."})
        gl.columns = list(gl.columns or []) + [labels[i] for tp in gl.templates or [] for i in tmap.get(tp, [])]
    if resp.layout is not None and "layout" not in out:
        out["layout"] = {"value": resp.layout.value, "confidence": _clamp(resp.layout.confidence),
                         "evidence": resp.layout.evidence, "source": "ai"}
        if resp.layout.value not in VOCABULARY["layout"]:
            out["layout"].update(value=UNRESOLVED, validation={
                "status": "contradicted", "messages": [f"'{resp.layout.value}' is not an allowed layout."]})
    groups = grouping.collect([(gl.group_id, gl.columns, _item(gl)) for gl in resp.groups],
                              set(indices), label_to_idx, ns, rej)
    for gl in resp.groups:
        if gl.suggest_split and f"{gl.group_id}{ns}" in groups:
            new = grouping.apply_split(groups, f"{gl.group_id}{ns}", gl.suggest_split, label_to_idx,
                                       lambda i, gl=gl: _split_item(gl, gl.evidence), rej)
            if new:
                out["splits_applied"].append({"group": gl.group_id, "new": new})
    for m in resp.propose_merge:
        keep = grouping.apply_merge(groups, [f"{g}{ns}" for g in m.group_ids], m.reason, rej)
        if keep:
            out["merges_applied"].append({"group_ids": m.group_ids, "into": keep, "reason": m.reason})
            for g in m.group_ids:
                out["alias"][f"{g}{ns}"] = keep
    out["groups"].update(groups)
    known = {a["assay_label"] for a in out["assays"]}
    for a in resp.assays:
        if (a.assay_label.strip() or "assay") in known:
            continue
        fi = a.feature_identity or FeatureIdentity()
        out["assays"].append({
            "assay_label": a.assay_label.strip() or "assay", "omics_type": a.omics_type or "unknown",
            "omics_family": a.omics_family if a.omics_family in VOCABULARY["omics_family"] else "unknown",
            "source_software": a.source_software or "unknown",
            "in_supported_scope": a.in_supported_scope if a.in_supported_scope in VOCABULARY["in_supported_scope"]
            else "unsure", "scope_reason": a.scope_reason or "",
            "feature_pids": [f"{g}{ns}" for g in fi.group_ids],
            "confidence": _clamp(a.confidence), "evidence": a.evidence, "source": "ai"})
    out["samples"] += [{"pattern_or_sample": x.pattern_or_sample, "label": x.label, "is_study_sample": x.is_study_sample,
                        "confidence": _clamp(x.confidence), "evidence": x.evidence, "source": "ai"} for x in resp.samples]
    out["clarifying_questions"] += [dict(q.model_dump(), group_id=f"{q.group_id}{ns}" if q.group_id else None)
                                    for q in resp.clarifying_questions]


def group_summary(cols, pid, g, chunk=None):
    """One compact line of facts per group, for the consolidation / merge-check call."""
    idx = sorted(g["indices"])
    names = [cols.labels[i] for i in idx]
    it = g["item"]
    st = [cols.digests[i] for i in idx]
    num = [d for d in st if d["type"] == "numeric"]
    meds = sorted(d.get("median") for d in num if d.get("median") is not None)
    pat = common_pattern([cols.header[i] for i in idx])
    stats = ("text" if not num else
             f"median of column medians {meds[len(meds) // 2] if meds else None}, "
             f"whole numbers {'yes' if all(d.get('integer_valued') for d in num) else 'no'}, "
             f"zeros {round(100 * sum(d.get('frac_zero') or 0 for d in num) / len(num))}%")
    from .profiling import name_template
    tpls = list(dict.fromkeys(name_template(n) for n in names))
    return {"group_id": pid, "chunk": chunk, "n_columns": len(idx), "role": it.get("role"), "label": it.get("label"),
            "assay_label": it.get("assay_label"), "audit_kind": it.get("audit_kind"), "family": it.get("family"),
            "marks_rows_as_suspect": it.get("marks_rows_as_suspect"), "first_columns": names[:2], "last_column": names[-1],
            "templates": tpls[:8] + ([f"... ({len(tpls) - 8} more)"] if len(tpls) > 8 else []),
            "shared_name_part": pat["text"] if pat else None, "stats": stats}


def normalize_item(item):
    """Drop fields that do not apply to the role (keeps the item consistent)."""
    role = item.get("role")
    if role != "value":
        if role not in ("feature_annotation",):
            item["assay_label"] = item.get("assay_label") if role == "feature_id" else None
    if role != "sample_metadata":
        item["audit_kind"] = None
    if role != "feature_annotation":
        item["marks_rows_as_suspect"] = False
    return item


def finalize_item(item, group, cols, layout=None):
    """Structural validation; contradicted -> unresolved (the claim is kept for display)."""
    item = normalize_item(item)
    v = validate_group(item, group, cols, layout)
    item["validation"] = v
    if v["status"] == "contradicted":
        item["claimed"] = {k: item.get(k) for k in ("role", "audit_kind", "label")}
        item["role"] = UNRESOLVED
    if item.get("role") == "sample_metadata" and item.get("audit_kind") == "timepoint" and group["n_columns"] == 1:
        if not item.get("detail"):
            item["detail"] = timepoint_detail(cols.digests[group["indices"][0]])
    return item


def unresolved_item(evidence, source="none"):
    return {"role": UNRESOLVED, "assay_label": None, "label": "", "audit_kind": None,
            "marks_rows_as_suspect": False, "confidence": 0.0, "evidence": evidence, "source": source,
            "validation": {"status": "ok", "messages": []}}


def plan_chunks(cols, affixes, indices, size):
    """(chunks, mode): each chunk is (column indices, name templates). Templates are the unit
    (v2.4 §15): a chunk holds at most `size` templates, so thousands of columns that compress to
    a handful of templates make one call. Unique, digit-free names fall back to column chunks."""
    tpls = name_templates(cols, indices)
    if len(tpls) <= size:
        return [(list(indices), tpls)], "templates"
    if all(t["n_columns"] == 1 for t in tpls):
        return [(ch, name_templates(cols, ch)) for ch in chunk_columns(indices, cols.labels, affixes, size)], "columns"
    parts = chunk_units([t["template"] for t in tpls], size)
    return [(sorted(i for k in part for i in tpls[k]["indices"]), [tpls[k] for k in part]) for part in parts], "templates"


def _final_item(fg):
    return {"role": fg.role, "assay_label": fg.assay_label, "label": (fg.label or "").strip(),
            "family": (fg.family or "").strip() or None, "audit_kind": fg.audit_kind,
            "marks_rows_as_suspect": bool(fg.marks_rows_as_suspect), "confidence": _clamp(fg.confidence),
            "evidence": fg.evidence, "source": "ai"}


def _assay_entry(a, ns="", map_fid=None):
    fi = a.feature_identity or FeatureIdentity()
    return {"assay_label": a.assay_label.strip() or "assay", "omics_type": a.omics_type or "unknown",
            "omics_family": a.omics_family if a.omics_family in VOCABULARY["omics_family"] else "unknown",
            "source_software": a.source_software or "unknown",
            "in_supported_scope": a.in_supported_scope if a.in_supported_scope in VOCABULARY["in_supported_scope"] else "unsure",
            "scope_reason": a.scope_reason or "",
            "feature_pids": [map_fid(g) if map_fid else f"{g}{ns}" for g in fi.group_ids],
            "confidence": _clamp(a.confidence), "evidence": a.evidence, "source": "ai"}


def apply_consolidation(cresp, out):
    """The final answer (v2.4 §15): final groups are unions of chunk groups ('members'); labels
    and assays come only from here. Chunk groups no final group names keep their columns but
    lose their draft label (unresolved, origin 'unconsolidated'); invalid members are rejected."""
    chunk_groups, rej = out["groups"], out["rejected"]
    final, claimed = {}, {}
    for fg in cresp.groups:
        valid = []
        for m in fg.members:
            m2 = out["alias"].get(m, m)
            if m2 not in chunk_groups:
                rej.append({"group_id": fg.group_id, "member": m, "reason": "The final answer names a chunk group "
                            "that does not exist; that member was rejected.", "where": "consolidation"})
            elif m2 in claimed:
                rej.append({"group_id": fg.group_id, "member": m, "reason": f"Chunk group already in final group "
                            f"{claimed[m2]}; the second claim was rejected.", "where": "consolidation"})
            else:
                valid.append(m2)
                claimed[m2] = fg.group_id
        pid = f"c:{fg.group_id}"
        if not valid or pid in final:
            rej.append({"group_id": fg.group_id, "reason": "The final group names no valid chunk group (or its id was "
                        "used twice); it was rejected.", "where": "consolidation"})
            continue
        final[pid] = {"indices": [i for m in valid for i in chunk_groups[m]["indices"]], "item": _final_item(fg),
                      "origin": "consolidated" if len(valid) == 1 else "merged_consolidation",
                      "merged_from": valid if len(valid) > 1 else None}
        if len(valid) > 1:
            out["merges_applied"].append({"group_ids": valid, "into": pid, "reason": fg.evidence, "cross_chunk": True})
    for m, g in chunk_groups.items():
        if m in claimed:
            continue
        if g["origin"] in ("ai_unavailable", "unmentioned") or g["item"].get("role") == UNRESOLVED:
            final[m] = g
        else:
            final[m] = {"indices": g["indices"], "origin": "unconsolidated", "item": dict(unresolved_item(
                "The final answer for the whole file did not place these columns: choose what they are."),
                claimed={k: g["item"].get(k) for k in ("role", "label")})}
    out["groups"] = final
    out["assays"] = [_assay_entry(a, map_fid=lambda g: f"c:{g}") for a in cresp.assays]
    out["clarifying_questions"] += [dict(q.model_dump(), group_id=f"c:{q.group_id}" if q.group_id else None)
                                    for q in cresp.clarifying_questions]
    out["alias"] = {}


def drafts_only(out):
    """The consolidation failed: chunk outputs are drafts, so their labels and assays are not
    used. Roles stay (the structure is still usable); every value block goes to one
    placeholder assay until the final answer is retried."""
    for g in out["groups"].values():
        it = g["item"]
        if it.get("role") != UNRESOLVED and it.get("source") == "ai":
            it.update(label="", evidence="Draft from one chunk; the final answer for the whole file failed. "
                                          "Retry it, or label this yourself.")
            if it.get("role") == "value":
                it["assay_label"] = "assay 1"
    fids = [p for a in out["assays"] for p in a.get("feature_pids", [])]
    out["assays"] = [{"assay_label": "assay 1", "omics_type": "unknown", "source_software": "unknown",
                      "in_supported_scope": "unsure", "scope_reason": "", "feature_pids": fids[:1],
                      "confidence": 0.0, "evidence": "Placeholder: the final answer for the whole file failed.",
                      "source": "none"}]


def propose(filename, sha, cols, affixes, hints, indices=None, fixed=None, log=None, on_progress=None,
            mock_fn=None, signature_hint=None, chunk_size=None):
    """Ask the AI to group and label the given columns (all by default).
    Name templates first; wide sets go in chunks of templates, then one final (consolidation)
    call that decides the final groups, labels and assays. Returns (proposal, meta, digests):
    proposal["groups"] is pid -> {"indices", "item", "origin"}, covering every column exactly once."""
    indices = list(range(len(cols.header))) if indices is None else sorted(indices)
    size = chunk_size or config.GROUPING_CHUNK_SIZE
    chunks, mode = plan_chunks(cols, affixes, indices, size)
    n = len(chunks)
    label_to_idx = {lab: i for i, lab in enumerate(cols.labels)}
    out = {"groups": {}, "rejected": [], "samples": [], "clarifying_questions": [], "assays": [],
           "splits_applied": [], "merges_applied": [], "alias": {}, "chunks": n, "chunk_mode": mode,
           "n_templates": sum(len(tp) for _, tp in chunks), "failed_chunks": [], "consolidation_error": None}
    metas, digests, chunk_of, chunk_assays = [], [], {}, []
    fixed = dict(fixed or {})
    total = n + (1 if n > 1 else 0)
    for k, (chunk, tpls) in enumerate(chunks, 1):
        ns = f"_chunk{k}" if n > 1 else ""
        if on_progress:
            on_progress(k - 1, total, f"AI call {k}/{total}: grouping and labelling {len(chunk)} column(s)"
                                      f" ({len(tpls)} name template(s))")
        digest = build_digest(filename, cols, affixes, hints, chunk, fixed, signature_hint, (k, n) if n > 1 else None, tpls)
        digests.append(digest)
        _ctx_set(on_progress, k, total)
        try:
            resp, meta = ask(digest, sha, log, mock_fn)
        finally:
            _ctx_clear()
        metas.append(meta)
        if resp is None:
            out["failed_chunks"].append(k)
            for i in chunk:
                out["groups"][f"u{i}{ns}"] = {"indices": [i], "origin": "ai_unavailable", "item": unresolved_item(
                    f"AI unavailable: {meta['error']}. Choose this column's role yourself.")}
            continue
        before = set(out["groups"])
        n_assays = len(out["assays"])
        read_chunk(resp, chunk, label_to_idx, ns, out, {tp["template"]: tp["indices"] for tp in tpls if tp["n_columns"] >= 2})
        chunk_assays += [dict(a, chunk=k) for a in out["assays"][n_assays:]]
        for pid in set(out["groups"]) - before:
            chunk_of[pid] = k
        if "layout" in out:
            fixed.setdefault("layout", out["layout"]["value"])
    if n > 1 and any(m.get("error") is None for m in metas):
        if on_progress:
            on_progress(n, total, f"AI call {total}/{total}: the final answer for the whole file ({n} chunks)")
        payload = {"groups": [group_summary(cols, pid, g, chunk_of.get(pid)) for pid, g in out["groups"].items()],
                   "chunk_assays": [{k: a.get(k) for k in ("chunk", "assay_label", "omics_type", "source_software")}
                                    for a in chunk_assays],
                   "already_confirmed": {"layout": fixed.get("layout")}}
        digests.append(payload)
        _ctx_set(on_progress, total, total)
        try:
            cresp, cmeta = _call(payload, sha, log, mock_fn, kind="consolidation")
        finally:
            _ctx_clear()
        metas.append(cmeta)
        if cresp is None:
            out["consolidation_error"] = cmeta.get("error") or "no answer"
            drafts_only(out)
        else:
            apply_consolidation(cresp, out)
    elif n > 1:
        out["consolidation_error"] = "every chunk failed"
    if log:
        log("grouping_result", {"chunks": n, "chunk_mode": mode, "n_templates": out["n_templates"],
                                "n_groups": len(out["groups"]), "merges": out["merges_applied"],
                                "splits": out["splits_applied"], "rejected": out["rejected"],
                                "consolidation_error": out["consolidation_error"]})
    if on_progress:
        on_progress(total, total, "AI proposal received and checked against the data")
    meta = metas[0] if metas else {}
    if any(m.get("error") for m in metas):
        meta = dict(meta, error=next(m["error"] for m in metas if m.get("error")))
    meta["models_used"] = sorted({m.get("model") for m in metas if m.get("model")})
    meta["calls"] = len(metas)
    meta["all_failed"] = bool(metas) and all(m.get("error") for m in metas[:n])
    return out, meta, digests


def consolidate(cols, groups, items, assays, sha, log=None, mock_fn=None):
    """The final-answer call on the current groups (e.g. to retry it after a failure).
    Returns (ConsolidationResponse | None, error). Nothing is applied here."""
    payload = {"groups": [group_summary(cols, gid, {"indices": g["indices"], "item": items[gid]})
                          for gid, g in groups.items()],
               "chunk_assays": [{"chunk": None, "assay_label": a["assay_label"], "omics_type": a.get("omics_type")}
                                for a in assays]}
    resp, meta = _call(payload, sha, log, mock_fn, kind="consolidation")
    if resp is None:
        return None, meta.get("error") or "no answer"
    return resp, None


def propose_metadata(filename, sha, cols, facts, data_ctx, fixed=None, log=None, mock_fn=None):
    """Sample metadata columns through the AI (v2.4 §6): digest + join facts, chunked when wide.
    Returns (merged answer dict | None, meta, digests). Nothing is applied here."""
    examples = send_examples()
    indices = list(range(len(cols.header)))
    size = config.GROUPING_CHUNK_SIZE
    chunks = [indices[k:k + size] for k in range(0, len(indices), size)] or [[]]
    merged = {"join_key": None, "columns": {}, "design": None, "questions": []}
    metas, digests = [], []
    for k, ch in enumerate(chunks, 1):
        cdig = []
        for i in ch:
            x = column_digest(cols, i, {}, examples)
            x.pop("shared_prefix", None)
            x.pop("shared_suffix", None)
            cdig.append(dict(x, file="metadata", join=facts[i]))
        payload = {"file": {"role": "metadata", "name_extension": Path(filename).suffix.lower(),
                            "n_rows": cols.n_rows, "n_columns": len(cols.header)},
                   "data": data_ctx, "join_facts_top": sorted(facts, key=lambda f: (-f["exact_matches"],
                                                                                   -f["normalized_matches"]))[:5],
                   "columns": cdig, "already_confirmed": fixed or {},
                   "settings": {"example_values_sent": examples, "raw_rows_sent": False}}
        if len(chunks) > 1:
            payload["chunk"] = {"index": k, "of": len(chunks)}
        digests.append(payload)
        resp, meta = _call(payload, sha, log, mock_fn, kind="metadata")
        metas.append(meta)
        if resp is None:
            continue
        if merged["join_key"] is None and resp.join_key and resp.join_key.get("column"):
            merged["join_key"] = resp.join_key
        names = {cols.labels[i] for i in ch}
        for c in resp.columns:
            if c.column in names and c.column not in merged["columns"]:
                merged["columns"][c.column] = c.model_dump()
        merged["design"] = merged["design"] or resp.design
        merged["questions"] += [q.model_dump() for q in resp.clarifying_questions]
    meta = metas[0] if metas else {}
    if any(m.get("error") for m in metas):
        meta = dict(meta, error=next(m["error"] for m in metas if m.get("error")))
    ok = any(m.get("error") is None for m in metas)
    return (merged if ok else None), meta, digests


def relabel(cols, affixes, indices, shared_label, sha, log=None, mock_fn=None):
    """One retry for annotation columns that all got the same label (v2.4 §8.1)."""
    examples = send_examples()
    payload = {"shared_label": shared_label, "instruction": "Give each column its own specific label.",
               "columns": [column_digest(cols, i, affixes[i], examples) for i in indices]}
    resp, meta = _call(payload, sha, log, mock_fn, kind="relabel")
    if resp is None:
        return None, meta.get("error")
    names = {cols.labels[i] for i in indices}
    return {x["column"]: x for x in resp.labels if x.get("column") in names and (x.get("label") or "").strip()}, None


def chat(payload, sha, log=None, mock_fn=None):
    """The user's message -> reply, proposed patches and questions (ChatResponse) or (None, error).
    Nothing is applied here."""
    resp, meta = _call(payload, sha, log, mock_fn, kind="chat")
    return resp, meta.get("error")


def audit_chat(payload, sha, log=None, mock_fn=None):
    """The audit operator: reply + tool calls (AuditChatResponse) or (None, error). Nothing runs here."""
    resp, meta = _call(payload, sha, log, mock_fn, kind="audit_chat")
    return resp, meta


def _ctx_set(on_progress, n, total):
    if on_progress:
        llm._ctx.report = lambda m: on_progress(n - 1, total, m)


def _ctx_clear():
    llm._ctx.report = None
