"""Orchestrator: Multi-Agent Pipeline Coordinator.

Wires all 5 agents together in sequence with strictly typed Pydantic contracts:
  1. ClaimCase input (tolerates unknown/extra fields)
  2. Agent 1 (Case Analysis): deterministic rule-based query planning & flags
  3. Agent 2 (Policy Evidence): deterministic hybrid retrieval & confidence signal
  4. Agent 3 (Coverage & Exclusion): LLM evaluation per bundle + code anti-hallucination guard
  5. Agent 4 (Decision): pure deterministic financial math + LLM narrative justification
  6. Agent 5 (Validation): LLM entailment re-check against raw chunks (downgrade to NEEDS_REVIEW if failed)
  7. CaseState output: unified trace
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.agents.case_analysis import analyze_case
from src.agents.coverage_exclusion import evaluate_all_bundles
from src.agents.decision import decide_claim
from src.agents.policy_evidence import gather_all_evidence
from src.agents.state import CaseState, ClaimCase, Decision
from src.agents.validation import validate_decision
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

# Shared retriever instance for efficiency across orchestrator calls
_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def run_case(raw_case_dict: dict[str, Any]) -> CaseState:
    """Execute the end-to-end multi-agent adjudication pipeline for a claim case."""
    # 1. Parse and validate input into typed ClaimCase (extra="allow")
    case = ClaimCase(**raw_case_dict)
    logger.info(f"[Orchestrator] Running claim adjudication for Case: {case.case_id}")

    # 2. Agent 1: Case Analysis (Deterministic, No LLM)
    plan = analyze_case(case)

    # 3. Agent 2: Policy Evidence Retrieval (Deterministic, No LLM)
    retriever = get_retriever()
    bundles, confidence_signal = gather_all_evidence(plan, retriever=retriever)

    # 4. Agent 3: Coverage and Exclusion Evaluation (LLM + Anti-Hallucination Guard)
    findings = evaluate_all_bundles(case, bundles)

    # 5. Agent 4: Adjudication & Financial Decision (Deterministic Math + LLM Narrative)
    raw_decision = decide_claim(
        case=case,
        plan=plan,
        findings=findings,
        confidence_signal=confidence_signal,
    )

    # 6. Agent 5: Validation & Entailment Re-Check (LLM Factual Guard)
    final_decision = validate_decision(raw_decision, bundles)

    return CaseState(
        case=case,
        plan=plan,
        evidence_bundles=bundles,
        findings=findings,
        decision=final_decision,
        confidence_signal=confidence_signal,
    )


if __name__ == "__main__":
    case_id = sys.argv[1] if len(sys.argv) > 1 else "PUB-001"
    cases_file = Path("data/cases/public_test_cases.json")
    with open(cases_file, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    match = next((c for c in all_cases if c["case_id"] == case_id), None)
    if not match:
        print(f"Case {case_id} not found in {cases_file}")
        sys.exit(1)

    state = run_case(match)
    dec = state.decision
    assert dec is not None

    print(f"\n========================================================")
    print(f"               CLAIM ADJUDICATION REPORT                ")
    print(f"========================================================")
    print(f"Case ID:         {state.case.case_id}")
    print(f"Diagnosis:       {state.case.treatment.get('diagnosis', 'N/A') if isinstance(state.case.treatment, dict) else state.case.treatment.diagnosis}")
    print(f"Claimed Amount:  INR {dec.claimed_amount_inr:,.2f}")
    print(f"Approved Amount: INR {dec.approved_amount_inr:,.2f}")
    print(f"Status:          {dec.status}")
    print(f"Confidence:      {state.confidence_signal:.2f}")

    if dec.deductions:
        print(f"\nItemized Deductions ({len(dec.deductions)}):")
        for d in dec.deductions:
            print(f"  - [{d.category.upper()}] Deducted: INR {d.deduction_inr:,.2f} | Reason: {d.reason}")
            print(f"    Policy Clause: {d.policy_clause}")

    if dec.missing_evidence:
        print(f"\nMissing Evidence / Abstention Signals:")
        for m in dec.missing_evidence:
            print(f"  ! {m}")

    print(f"\nJustification Narrative:")
    print(dec.justification)

    print(f"\nValidation Notes ({len(dec.validation_notes)}):")
    for vn in dec.validation_notes:
        print(f"  * {vn}")
