from enum import Enum
from typing import List, Tuple, Optional, Literal
from pydantic import BaseModel

class EvidenceStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTORY = "CONTRADICTORY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NEEDS_REVIEW = "NEEDS_REVIEW"

class FailureCode(str, Enum):
    ERR_UNSUPPORTED = "ERR_UNSUPPORTED"
    ERR_MISINTERPRETATION = "ERR_MISINTERPRETATION"
    ERR_INCOMPLETE = "ERR_INCOMPLETE"
    ERR_TONE_MISMATCH = "ERR_TONE_MISMATCH"
    ERR_TRACEABILITY = "ERR_TRACEABILITY"

class RunStatus(str, Enum):
    COMPLETED = "completed"
    EVALUATION_FAILED = "EVALUATION_FAILED"

class EvidenceRecord(BaseModel):
    source_id: str
    record_id: str
    text: str
    meta: dict = {}

class Anchor(BaseModel):
    source_id: str
    record_id: str
    excerpt: str
    valid: Optional[bool] = None

class Suggestion(BaseModel):
    action_type: str
    text: Optional[str]
    grounded_in: List[Anchor]
    limit_note: str
    user_decision: Literal["pending", "accepted", "edited", "ignored"] = "pending"

class Claim(BaseModel):
    claim_id: str
    text: str
    span: Tuple[int, int]
    evidence_status: EvidenceStatus
    reason: str
    failure_codes: List[FailureCode] = []
    evidence_anchors: List[Anchor] = []
    suggestion: Optional[Suggestion] = None

class ChecklistItem(BaseModel):
    field: str
    status: Literal["present", "missing_source_has", "missing_source_lacks", "n/a"]
    description: str

class ToneWarning(BaseModel):
    phrase: str
    reason: str
    severity: str

class EditingMetrics(BaseModel):
    edit_ratio: float
    active_review_seconds: float

class Evaluation(BaseModel):
    schema_version: str = "1.0"
    case_id: str
    session_id: str
    evaluation_id: str
    article_version: int
    rubric_version: str
    model: str
    claims: List[Claim]
    completeness: List[ChecklistItem]
    tone_warnings: List[ToneWarning]
    editing: Optional[EditingMetrics]
    latency_s: float
    cost_usd: Optional[float] = None
    run_status: RunStatus
