"""Unit and integration tests for Multi-Agent Health Insurance Claim Engine."""
import json
from pathlib import Path
import pytest
from src.agents.state import (
    ClaimCase,
    Finding,
    EvidenceBundle,
    Decision,
)
from src.agents.decision import calculate_decision_math
from src.agents.validation import validate_decision
from src.retrieval.dense import RetrievedChunk


def get_sample_case(case_id: str = "PUB-001") -> ClaimCase:
    cases_file = Path("data/cases/public_test_cases.json")
    with open(cases_file, "r", encoding="utf-8") as f:
        cases = json.load(f)
    match = next(c for c in cases if c["case_id"] == case_id)
    return ClaimCase(**match)


def test_claim_case_math_validation():
    """Verify ClaimCase validates claimed total."""
    case = get_sample_case("PUB-001")
    assert case.get_claimed_total() == 163200.0
    assert case.get_days_admitted() == 4


def test_decision_math_sublimits_room_and_ambulance():
    """Verify deterministic math caps room rent to 1% BSI/day and ambulance to min(1% BSI, 1000)."""
    case = get_sample_case("PUB-001")
    findings = [
        Finding(
            issue="base_coverage",
            applies="yes",
            citation_chunk_id="what-we-cover-00",
            claim="Hospitalisation covered up to Sum Insured.",
        ),
        Finding(
            issue="room_rent_sub_limit",
            applies="yes",
            citation_chunk_id="sub-limits-00",
            claim="Normal room capped at 1.0% Basic Sum Insured per day.",
        ),
        Finding(
            issue="ambulance_sub_limit",
            applies="yes",
            citation_chunk_id="sub-limits-item-4-00",
            claim="Ambulance capped at lesser of 1% BSI or INR 1,000.",
        ),
    ]

    status, claimed, approved, deductions, missing = calculate_decision_math(
        case=case,
        plan=None,
        findings=findings,
        confidence_signal=1.0,
    )

    assert status == "PARTIALLY_APPROVED"
    assert claimed == 163200.0
    assert approved == 153000.0
    assert len(deductions) == 2
    ded_dict = {d.category: d.deduction_inr for d in deductions}
    assert ded_dict["room"] == 10000.0
    assert ded_dict["ambulance"] == 200.0


def test_decision_math_waiting_period_rejection():
    """Verify that initial 30-day waiting period rejects claim completely."""
    case = get_sample_case("PUB-002")
    findings = [
        Finding(
            issue="initial_waiting_period",
            applies="yes",
            citation_chunk_id="clause-2-00",
            claim="30 days waiting period applies to all illnesses.",
        )
    ]

    status, claimed, approved, deductions, missing = calculate_decision_math(
        case=case,
        plan=None,
        findings=findings,
        confidence_signal=1.0,
    )

    assert status == "REJECTED"
    assert approved == 0.0
    assert len(deductions) == 1
    assert deductions[0].deduction_inr == claimed


def test_decision_math_pre_existing_rejection():
    """Verify that pre-existing condition with < 48 months coverage is rejected."""
    case = get_sample_case("PUB-003")
    findings = [
        Finding(
            issue="pre_existing",
            applies="yes",
            citation_chunk_id="clause-1-00",
            claim="48 months waiting period applies to pre-existing conditions.",
        )
    ]

    status, claimed, approved, deductions, missing = calculate_decision_math(
        case=case,
        plan=None,
        findings=findings,
        confidence_signal=1.0,
    )

    assert status == "REJECTED"
    assert approved == 0.0


def test_anti_hallucination_guard_missing_chunk():
    """Agent 5 must downgrade to NEEDS_REVIEW if a cited chunk_id does not exist in the evidence set."""
    fake_finding = Finding(
        issue="room_rent_sub_limit",
        applies="yes",
        citation_chunk_id="hallucinated-chunk-999",  # Does not exist in pool
        claim="Invented clause text.",
    )

    initial_dec = Decision(
        status="PARTIALLY_APPROVED",
        claimed_amount_inr=100000.0,
        approved_amount_inr=90000.0,
        deductions=[],
        justification="Initial test justification.",
        supporting_findings=[fake_finding],
        missing_evidence=[],
        validation_notes=[],
    )

    from src.agents.state import EvidenceChunk

    evidence_bundles = [
        EvidenceBundle(
            issue="room_rent_sub_limit",
            query="test query",
            rationale="test rationale",
            chunks=[
                EvidenceChunk(
                    chunk_id="real-chunk-01",
                    text="Actual policy text here.",
                    section="Sub limits",
                    page_start=7,
                    page_end=8,
                    dense_rank=1,
                    sparse_rank=1,
                    fused_score=0.015,
                )
            ],
        )
    ]

    validated_dec = validate_decision(initial_dec, evidence_bundles)
    assert validated_dec.status == "NEEDS_REVIEW"
    assert any("not found in evidence pool" in note for note in validated_dec.validation_notes)
