"""Request / response schemas for the PRISM Step 0 API."""

from __future__ import annotations

from typing import Dict, List, Optional

from typing_extensions import Literal  # typing.Literal needs Python 3.8+

from pydantic import BaseModel

# Fixed role vocabulary. The AI may additionally answer "unresolved", which is
# never accepted by the confirmation endpoint: a human must pick a real role.
ROLES = (
    "sample_id",
    "subject_id",
    "timepoint",
    "batch",
    "group_or_outcome",
    "feature_value",
    "feature_annotation",
    "ignore",
)
Role = Literal[
    "sample_id",
    "subject_id",
    "timepoint",
    "batch",
    "group_or_outcome",
    "feature_value",
    "feature_annotation",
    "ignore",
]
UNRESOLVED = "unresolved"


class ColumnProposal(BaseModel):
    """One AI proposal, exactly the shape the model is asked to return."""

    column: str
    proposed_role: str
    confidence: float
    evidence: str


class AIProposalBatch(BaseModel):
    """Structured-output schema handed to the LLM."""

    columns: List[ColumnProposal]


class ColumnRole(BaseModel):
    index: int
    column: str
    role: str
    confidence: Optional[float] = None
    evidence: Optional[str] = None


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    delimiter: str
    encoding: str
    n_rows: int
    n_columns: int
    header: List[str]
    preview_rows: List[List[str]]
    warnings: List[str]
    method: Literal["signature", "ai", "manual"]
    detection: Optional[dict]
    ai: Optional[dict]
    columns: List[ColumnRole]
    roles_vocabulary: List[str]


class ConfirmedColumn(BaseModel):
    index: int
    column: str
    role: Role


class ConfirmRequest(BaseModel):
    columns: List[ConfirmedColumn]


class StructureSummary(BaseModel):
    session_id: str
    confirmed_at: str
    filename: str
    method: str
    platform: str
    omics_type: str
    layout: str
    layout_description: str
    sample_count: Optional[int]
    feature_count: Optional[int]
    feature_id: str
    n_rows: int
    n_columns: int
    role_counts: Dict[str, int]
    design_factors: List[dict]
    warnings: List[str]
    columns: List[dict]
    changes_from_proposal: int
    log_file: str
