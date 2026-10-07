"""The schema.json contract (0.4.x) as Pydantic models (v3 §4.2).

The published JSON Schema (schema/prism_schema_0.4.json) is generated from these models
(python -m prism.io.schema_model). Unknown fields are kept (extra='allow') and reported as
warnings; closed fields use the Step 0 vocabularies. Nothing in a schema is ever executed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

try:
    from typing import Literal
except ImportError:   # Python 3.7
    from typing_extensions import Literal

from backend.schema import VOCABULARY

SUPPORTED = re.compile(r"^0\.4\.\d+$")
MAX_BYTES = 20 * 1024 * 1024
JSON_SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "schema" / "prism_schema_0.4.json"

OmicsFamilyValue = Literal[tuple(VOCABULARY["omics_family"])]
AuditKind = Literal[tuple(VOCABULARY["audit_kind"])]
Layout = Literal[tuple(VOCABULARY["layout"])]
Provenance = Literal["computed", "ai_proposed_confirmed", "ai_proposed_corrected", "user_set", "unanswered"]


class Open(BaseModel):
    model_config = ConfigDict(extra="allow")


class Fact(Open):
    value: Layout
    provenance: Provenance


class FeatureIdentity(Open):
    columns: List[str] = []
    from_: Optional[str] = Field(default=None, alias="from")
    composite: bool = False
    provenance: str = "computed"


class SampleIdRule(Open):
    strip_prefix: str = ""
    strip_suffix: str = ""
    add_prefix: str = ""


class ValueBlock(Open):
    block_id: str
    group_id: str
    columns: List[str] = Field(min_length=1)
    label: str = ""
    keep: bool = True
    confidence: Optional[float] = None
    provenance: Provenance
    sample_id_rule: Optional[SampleIdRule] = None
    profile: Dict[str, Any] = {}
    feature_facts: Optional[Dict[str, Any]] = None
    n_features: int
    n_samples: int
    file: Optional[str] = None   # ignored on import: outputs are regenerated


class OmicsFamily(Open):
    value: OmicsFamilyValue
    provenance: str


class Assay(Open):
    assay_id: str
    assay_label: str
    omics_type: str = "unknown"
    omics_family: Optional[OmicsFamily] = None
    source_software: str = "unknown"
    in_supported_scope: Literal["yes", "no", "unsure"] = "unsure"
    scope_reason: str = ""
    provenance: Provenance
    feature_identity: FeatureIdentity
    value_blocks: List[ValueBlock] = Field(min_length=1)
    n_features: int
    n_samples: int
    n_value_columns: int


class FeatureAnnotation(Open):
    column: str
    group_id: Optional[str] = None
    label: str = ""
    is_feature_id: Optional[bool] = None
    marks_rows_as_suspect: Optional[bool] = None
    keep: bool = True
    provenance: str
    derived_from: Optional[str] = None
    rule: Optional[Dict[str, Any]] = None
    family: Optional[str] = None
    flag_values: Optional[Dict[str, int]] = None
    flagged_values: Optional[List[str]] = None
    flag_counts: Optional[Dict[str, int]] = None
    n_flagged: Optional[int] = None
    source: Optional[str] = None
    from_column: Optional[str] = None
    display_labels: Optional[Dict[str, str]] = None
    coverage: Optional[float] = None


class SampleMetadataColumn(Open):
    column: str
    group_id: Optional[str] = None
    audit_kind: Optional[AuditKind] = None
    label: str = ""
    keep: bool = True
    file: Optional[str] = None
    source: Optional[str] = None
    provenance: str
    detail: Optional[str] = None
    family: Optional[str] = None
    confidence: Optional[float] = None
    evidence: Optional[str] = None
    varies_within_subject: Optional[Union[bool, str]] = None
    derived_from: Optional[str] = None
    rule: Optional[Dict[str, Any]] = None


class DesignSide(Open):
    source: Literal["metadata_column", "derived_from_sample_names", "none"]
    provenance: str
    column: Optional[str] = None
    file: Optional[str] = None
    derivation: Optional[Dict[str, Any]] = None
    n_subjects: Optional[int] = None
    kind: Optional[str] = None
    n_distinct: Optional[int] = None
    unit: Optional[Dict[str, Any]] = None


class Design(Open):
    subject: DesignSide
    time: DesignSide
    repeated_measures: Dict[str, Any] = {}
    cross_checks: List[Dict[str, Any]] = []
    label: Optional[str] = None


class Sample(Open):
    sample: str
    label: str = ""
    is_study_sample: bool = True
    provenance: str


class Excluded(Open):
    column: str
    file: str = "main"
    reason: str
    by: str
    at: Optional[str] = None
    role: Optional[str] = None
    group_id: Optional[str] = None


class FileEntry(Open):
    file_role: Literal["main", "metadata"]
    name: str
    sha256: Optional[str] = None
    n_rows: Optional[int] = None
    n_columns: Optional[int] = None
    source_columns: Optional[List[str]] = None
    source_columns_sha256: Optional[str] = None
    parse_report: Dict[str, Any] = {}
    join: Optional[Dict[str, Any]] = None


class HistoryItem(Open):
    answer: Optional[Literal["yes", "no", "not_sure"]] = None
    note: str = ""
    provenance: Provenance = "unanswered"
    answered_at: Optional[str] = None


class Question(Open):
    question_id: str
    source: str
    type: str
    text: str
    status: Literal["open", "answered", "dismissed"]
    applies_to: Dict[str, Any] = {}
    answer: Optional[Dict[str, Any]] = None
    kind: Optional[str] = None
    key: Optional[str] = None
    step: Optional[str] = None
    options: List[Dict[str, Any]] = []


class SchemaDoc(Open):
    schema_version: str
    source_file: str
    file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    layout: Fact
    omics_family: Optional[Dict[str, Any]] = None
    assays: List[Assay] = Field(min_length=1)
    feature_annotations: List[FeatureAnnotation] = []
    sample_metadata: List[SampleMetadataColumn] = []
    design: Optional[Design] = None
    sample_id: Dict[str, Any]
    samples: List[Sample] = []
    excluded_columns: List[Excluded] = []
    files: List[FileEntry] = Field(min_length=1)
    column_ledger: Dict[str, Any] = {}
    processing_history: Dict[str, Union[HistoryItem, str]] = {}
    parse_report: Dict[str, Any] = {}
    integrity_flags: List[Dict[str, Any]] = []
    ai: Optional[Dict[str, Any]] = None
    signature_hint: Optional[str] = None
    questions: List[Question] = []
    log_ref: Optional[str] = None


class SchemaRejected(Exception):
    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors or []


def _path(loc):
    out = "$"
    for x in loc:
        out += f"[{x}]" if isinstance(x, int) else f".{x}"
    return out


def _extras(obj, path="$", out=None):
    out = [] if out is None else out
    if isinstance(obj, BaseModel):
        for k in (obj.model_extra or {}):
            out.append(f"{path}.{k}")
        for name in type(obj).model_fields:
            _extras(getattr(obj, name), f"{path}.{name}", out)
    elif isinstance(obj, list):
        for i, x in enumerate(obj):
            _extras(x, f"{path}[{i}]", out)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, BaseModel):
                _extras(v, f"{path}.{k}", out)
    return out


KNOWN_EXTRAS = {"$.omics_family"}


def validate(raw_bytes):
    """-> (schema dict, warnings). Raises SchemaRejected with [{path, message}] errors."""
    if len(raw_bytes) > MAX_BYTES:
        raise SchemaRejected("The schema file is larger than 20 MB.")
    try:
        doc = json.loads(raw_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as e:
        raise SchemaRejected(f"schema.json is not valid JSON: {e}")
    if not isinstance(doc, dict):
        raise SchemaRejected("schema.json must be a JSON object.")
    ver = str(doc.get("schema_version") or "")
    if not SUPPORTED.match(ver):
        raise SchemaRejected(f"Schema version '{ver or 'missing'}' is not supported (0.4.x only): re-run Step 0 on "
                             "this file to make a current schema. Old schemas are never migrated silently.")
    try:
        model = SchemaDoc.model_validate(doc)
    except ValidationError as e:
        errs = [{"path": _path(x["loc"]), "message": x["msg"]} for x in e.errors()]
        raise SchemaRejected(f"schema.json does not match the 0.4 contract ({len(errs)} problem(s)).", errs)
    warnings = [f"unknown field kept: {p}" for p in _extras(model) if p not in KNOWN_EXTRAS]
    return doc, warnings


def json_schema():
    s = SchemaDoc.model_json_schema()
    s["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    s["$id"] = "https://prism/schema/prism_schema_0.4.json"
    s["title"] = "PRISM Step 0 schema.json (0.4.x)"
    return s


def write_json_schema(path=JSON_SCHEMA_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(json_schema(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(write_json_schema())
