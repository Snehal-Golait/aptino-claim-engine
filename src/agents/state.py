"""Shared, strictly typed Pydantic state threaded through the multi-agent pipeline.

Enforces genuine multi-agent separation: each agent operates on typed
contracts with explicit boundaries, preventing prompts from collapsing into
one monolithic call or leaking unvetted context into the decision phase.
"""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class PatientInfo(BaseModel):
    model_config = ConfigDict(extra="allow")
    age: int | None = None
    gender: str | None = None
    relationship: str | None = None


class HospitalInfo(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    network_provider: bool | None = None
    address: str | None = None


class TreatmentInfo(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str  # e.g. "inpatient", "day_care", "domiciliary"
    admission_hours: int = 0
    diagnosis: str
    procedure: str
    pre_existing: bool = False
    experimental: bool = False
    hospital_room_unavailable: bool | None = None
    patient_cannot_be_moved: bool | None = None


class ExpensesInfo(BaseModel):
    model_config = ConfigDict(extra="allow")
    room: float = 0.0
    doctor_fees: float = 0.0
    medicines_diagnostics: float = 0.0
    pre_hospitalization: float = 0.0
    post_hospitalization: float = 0.0
    ambulance: float = 0.0


class ExpenseTimingInfo(BaseModel):
    model_config = ConfigDict(extra="allow")
    pre_hospitalization_days_before_admission: int | None = None
    post_hospitalization_days_after_discharge: int | None = None
    same_condition_confirmed: bool | None = None


class PriorPolicyInfo(BaseModel):
    model_config = ConfigDict(extra="allow")
    insurer_type: str | None = None
    continuous_years: int | float = 0
    database_and_claim_history_received: bool | None = None
    previous_sum_insured_inr: float | None = None


class ClaimCase(BaseModel):
    """Input claim case.

    extra='allow' is critical: the take-home schema explicitly requires
    tolerating unknown / extra fields without failure.
    """
    model_config = ConfigDict(extra="allow")

    case_id: str
    policy_id: str
    policy_start_date: str
    claim_date: str
    sum_insured_inr: float
    continuous_coverage_months: int | float = 0
    prior_insurer_continuous_years: int | float = 0
    patient: PatientInfo | dict[str, Any] = Field(default_factory=dict)
    hospital: HospitalInfo | dict[str, Any] = Field(default_factory=dict)
    treatment: TreatmentInfo | dict[str, Any]
    expenses_inr: ExpensesInfo | dict[str, float]
    documents: list[str] = Field(default_factory=list)
    task: str = ""

    # Optional fields
    evidence_context: dict[str, Any] | None = None
    expense_timing: ExpenseTimingInfo | dict[str, Any] | None = None
    prior_policy: PriorPolicyInfo | dict[str, Any] | None = None

    def get_expenses_dict(self) -> dict[str, float]:
        if isinstance(self.expenses_inr, ExpensesInfo):
            return self.expenses_inr.model_dump()
        return dict(self.expenses_inr)

    def get_treatment_dict(self) -> dict[str, Any]:
        if isinstance(self.treatment, TreatmentInfo):
            return self.treatment.model_dump()
        return dict(self.treatment)

    def get_claimed_total(self) -> float:
        exp = self.get_expenses_dict()
        return float(sum(exp.values()))

    def get_days_admitted(self) -> int:
        treatment = self.get_treatment_dict()
        hours = treatment.get("admission_hours", 0)
        return max(1, int(round(hours / 24)))


class RetrievalQuery(BaseModel):
    issue: str
    query: str
    rationale: str


class RetrievalPlan(BaseModel):
    case_id: str
    queries: list[RetrievalQuery]
    flags: dict[str, Any] = Field(default_factory=dict)


class EvidenceChunk(BaseModel):
    chunk_id: str
    section: str
    page_start: int
    page_end: int
    text: str
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused_score: float | None = None
    rerank_score: float | None = None


class EvidenceBundle(BaseModel):
    issue: str
    query: str
    rationale: str
    chunks: list[EvidenceChunk] = Field(default_factory=list)
    max_confidence: float = 0.0


class Finding(BaseModel):
    """Output of Agent 3 (Coverage & Exclusion).

    applies: 'yes' (clause applies to the claim),
             'no' (clause does not apply / condition cleared),
             'uncertain' (insufficient evidence or ambiguous excerpt).
    """
    issue: str
    applies: Literal["yes", "no", "uncertain"]
    citation_chunk_id: str | None = None
    citation_section: str | None = None
    citation_pages: str | None = None
    claim: str
    reasoning: str = ""


class LineItemDeduction(BaseModel):
    category: str
    claimed_inr: float
    deduction_inr: float
    sublimit_applied_inr: float | None = None
    reason: str
    policy_clause: str | None = None


class Decision(BaseModel):
    """Output of Agent 4 (Decision) & Agent 5 (Validation)."""
    status: Literal["APPROVED", "PARTIALLY_APPROVED", "REJECTED", "NEEDS_REVIEW"]
    claimed_amount_inr: float
    approved_amount_inr: float
    deductions: list[LineItemDeduction] = Field(default_factory=list)
    justification: str
    supporting_findings: list[Finding] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    validation_notes: list[str] = Field(default_factory=list)


class CaseState(BaseModel):
    """Unified state threaded through all 5 pipeline agents."""
    case: ClaimCase
    plan: RetrievalPlan | None = None
    evidence_bundles: list[EvidenceBundle] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    decision: Decision | None = None
    confidence_signal: float = 0.0
