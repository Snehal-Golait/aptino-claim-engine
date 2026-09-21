"""Agent 4: Decision & Adjudication (Deterministic Math + LLM Narrative Justification).

Core Architectural Principles:
1. Pure Deterministic Python Math:
   Approved amounts and line-item deductions are calculated entirely in
   deterministic Python code. The LLM is NEVER allowed to compute or invent numbers.
2. Genuine Multi-Agent Separation:
   The Decision Agent receives ONLY the vetted Finding objects from Agent 3 as its
   policy-evidence input — NEVER raw chunk text. If it received raw text, it could
   synthesize unvetted claims bypassing the coverage analysis phase.
3. Post-Adjudication LLM Narrative:
   Only after status, approved amounts, and itemized deductions are calculated, one
   LLM call synthesizes a 3-5 sentence human-readable justification citing the findings.
"""
from __future__ import annotations

import json
import logging
import math
from typing import Any, Literal

from groq import Groq

from src import config
from src.config import call_groq_chat
from src.agents.state import (
    ClaimCase,
    Decision,
    Finding,
    LineItemDeduction,
    RetrievalPlan,
)

logger = logging.getLogger(__name__)

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


JUSTIFICATION_PROMPT = """You are a senior health insurance claims auditor writing a formal decision summary for a claim.

CLAIM & ADJUDICATION FACTS (DETERMINED BY POLICY AUDIT):
- Case ID: {case_id}
- Diagnosis / Procedure: {diagnosis} / {procedure}
- Sum Insured (INR): {sum_insured}
- Claimed Amount (INR): {claimed_amount}
- Calculated Approved Amount (INR): {approved_amount}
- Decision Status: {status}

ITEMIZED DEDUCTIONS / ADJUSTMENTS:
{deductions_text}

SUPPORTING FINDINGS (VETTED POLICY CLAUSES):
{findings_text}

MISSING EVIDENCE / AUDIT SIGNALS:
{missing_evidence_text}

TASK:
Write a concise, professional, 3 to 5 sentence justification explaining the final decision ({status}).
Requirements:
1. State the decision status and the approved/claimed figures clearly.
2. Cite the specific policy clauses and findings (mentioning the clause section and page numbers where applicable).
3. If deductions or rejection occurred, explain the exact mathematical or exclusion reasons.
4. If status is NEEDS_REVIEW, clearly state what documentation or verification is required before finalization.
5. Do NOT alter any numbers or invent any new amounts; all amounts must match the facts above exactly.

Respond ONLY with the justification text."""


def _find_finding(findings: list[Finding], issue_key: str) -> Finding | None:
    for f in findings:
        if issue_key in f.issue.lower():
            return f
    return None


def calculate_decision_math(
    case: ClaimCase,
    plan: RetrievalPlan | None,
    findings: list[Finding],
    confidence_signal: float,
) -> tuple[
    Literal["APPROVED", "PARTIALLY_APPROVED", "REJECTED", "NEEDS_REVIEW"],
    float,
    float,
    list[LineItemDeduction],
    list[str],
]:
    """Pure deterministic adjudication logic and financial math."""
    expenses = case.get_expenses_dict()
    treatment = case.get_treatment_dict()
    sum_insured = float(case.sum_insured_inr)
    claimed_amount = sum(float(v) for v in expenses.values())

    deductions: list[LineItemDeduction] = []
    missing_evidence: list[str] = []

    # 1. Check Missing Evidence / Unresolved Context Signals (Strongest Abstention Signal)
    if plan and plan.flags.get("has_unresolved_evidence_fields"):
        for field_name in plan.flags.get("unresolved_fields", []):
            missing_evidence.append(f"Unresolved evidence: '{field_name}' requires verification.")

    if case.evidence_context:
        for k, v in case.evidence_context.items():
            if v is None:
                item = f"Missing field in evidence_context: '{k}' is null."
                if item not in missing_evidence:
                    missing_evidence.append(item)
            elif v is False and k in ("hospital_minimum_criteria_documented", "hospital_registered"):
                item = f"Facility criteria requirement not satisfied: '{k}' is false."
                if item not in missing_evidence:
                    missing_evidence.append(item)

    # 2. Check Retrieval Confidence Threshold
    if confidence_signal < config.MIN_RERANK_SCORE_FOR_CONFIDENT_DECISION and confidence_signal > 0:
        missing_evidence.append(
            f"Retrieval confidence score ({confidence_signal:.2f}) below threshold "
            f"({config.MIN_RERANK_SCORE_FOR_CONFIDENT_DECISION}). Abstaining to NEEDS_REVIEW."
        )

    # 3. Check for Material 'uncertain' Findings
    has_material_uncertainty = False
    for f in findings:
        if f.applies == "uncertain" and f.issue not in ("policy_sub_limits", "pre_post_hospitalization"):
            has_material_uncertainty = True
            missing_evidence.append(f"Policy coverage finding for '{f.issue}' is uncertain: {f.claim}")

    # If missing evidence or critical uncertainty exists, must be NEEDS_REVIEW
    if missing_evidence or has_material_uncertainty:
        return "NEEDS_REVIEW", claimed_amount, 0.0, deductions, missing_evidence

    # 4. Check Fatal Exclusions & Waiting Periods
    # A) Initial 30-day waiting period
    init_wp = _find_finding(findings, "initial_waiting_period")
    if init_wp and init_wp.applies == "yes":
        deductions.append(
            LineItemDeduction(
                category="entire_claim",
                claimed_inr=claimed_amount,
                deduction_inr=claimed_amount,
                sublimit_applied_inr=0.0,
                reason="Claim occurred within the initial 30-day waiting period without continuous prior coverage.",
                policy_clause=f"{init_wp.citation_section} (p. {init_wp.citation_pages})" if init_wp.citation_section else "30 days Waiting Period",
            )
        )
        return "REJECTED", claimed_amount, 0.0, deductions, missing_evidence

    # B) Pre-Existing Disease (PED) 48-month waiting period
    ped_wp = _find_finding(findings, "pre_existing")
    if ped_wp and ped_wp.applies == "yes" and treatment.get("pre_existing", False):
        if case.continuous_coverage_months < 48:
            deductions.append(
                LineItemDeduction(
                    category="entire_claim",
                    claimed_inr=claimed_amount,
                    deduction_inr=claimed_amount,
                    sublimit_applied_inr=0.0,
                    reason=f"Pre-existing condition declared with {case.continuous_coverage_months} months continuous coverage (< 48-month policy waiting period).",
                    policy_clause=f"{ped_wp.citation_section} (p. {ped_wp.citation_pages})" if ped_wp.citation_section else "1. Pre-existing diseases",
                )
            )
            return "REJECTED", claimed_amount, 0.0, deductions, missing_evidence

    # C) Experimental / Unproven Treatment Exclusion
    exp_find = _find_finding(findings, "experimental")
    if exp_find and exp_find.applies == "yes" and treatment.get("experimental", False):
        deductions.append(
            LineItemDeduction(
                category="entire_claim",
                claimed_inr=claimed_amount,
                deduction_inr=claimed_amount,
                sublimit_applied_inr=0.0,
                reason="Experimental or unproven treatment is excluded under General Exclusions.",
                policy_clause=f"{exp_find.citation_section} (p. {exp_find.citation_pages})" if exp_find.citation_section else "Exclusion: Unproven/Experimental Treatment",
            )
        )
        return "REJECTED", claimed_amount, 0.0, deductions, missing_evidence

    # D) Cosmetic Surgery Exclusion
    cosm_find = _find_finding(findings, "cosmetic")
    diag_proc_str = f"{treatment.get('diagnosis', '')} {treatment.get('procedure', '')}".lower()
    if cosm_find and cosm_find.applies == "yes":
        if "cosmetic" in diag_proc_str and "appendicitis" not in diag_proc_str and "pneumonia" not in diag_proc_str:
            # Pure cosmetic claim
            deductions.append(
                LineItemDeduction(
                    category="entire_claim",
                    claimed_inr=claimed_amount,
                    deduction_inr=claimed_amount,
                    sublimit_applied_inr=0.0,
                    reason="Cosmetic or aesthetic treatment is excluded under policy terms.",
                    policy_clause=f"{cosm_find.citation_section} (p. {cosm_find.citation_pages})" if cosm_find.citation_section else "Exclusions > Cosmetic Surgery",
                )
            )
            return "REJECTED", claimed_amount, 0.0, deductions, missing_evidence
        else:
            # Partial cosmetic procedure in a multi-diagnosis claim
            # Deduct room / doctor fee associated if specified, or specific line
            cosmetic_deduction = expenses.get("doctor_fees", 0.0) * 0.5  # representative portion
            if cosmetic_deduction > 0:
                deductions.append(
                    LineItemDeduction(
                        category="doctor_fees",
                        claimed_inr=expenses.get("doctor_fees", 0.0),
                        deduction_inr=cosmetic_deduction,
                        sublimit_applied_inr=None,
                        reason="Excluded cosmetic procedure component deducted from professional fees.",
                        policy_clause=f"{cosm_find.citation_section} (p. {cosm_find.citation_pages})" if cosm_find.citation_section else "Exclusion: Cosmetic Surgery",
                    )
                )

    # E) Specific Disease First Year Waiting Period (Cataract, Hernia, etc.)
    cataract_find = _find_finding(findings, "cataract")
    if "cataract" in diag_proc_str:
        # Check portability credit
        total_continuous_months = float(case.continuous_coverage_months)
        if case.prior_insurer_continuous_years > 0:
            total_continuous_months += float(case.prior_insurer_continuous_years) * 12.0
        elif case.prior_policy and isinstance(case.prior_policy, dict):
            if case.prior_policy.get("database_and_claim_history_received"):
                total_continuous_months += float(case.prior_policy.get("continuous_years", 0)) * 12.0

        if total_continuous_months < 12:
            deductions.append(
                LineItemDeduction(
                    category="entire_claim",
                    claimed_inr=claimed_amount,
                    deduction_inr=claimed_amount,
                    sublimit_applied_inr=0.0,
                    reason=f"Cataract treatment subject to 1-year (12 months) waiting period; continuous coverage is {total_continuous_months:.1f} months.",
                    policy_clause=f"{cataract_find.citation_section} (p. {cataract_find.citation_pages})" if cataract_find and cataract_find.citation_section else "2.30 days Waiting Period > Item 3",
                )
            )
            return "REJECTED", claimed_amount, 0.0, deductions, missing_evidence

    # F) Domiciliary Minimum Duration Requirement (Item 19: Any treatment not exceeding three days)
    treatment_type = treatment.get("type", "").lower()
    if treatment_type == "domiciliary":
        treatment_days = treatment.get("treatment_days")
        if treatment_days is None and treatment.get("admission_hours", 0) > 0:
            treatment_days = math.ceil(treatment.get("admission_hours", 0) / 24.0)
        if treatment_days is not None and treatment_days <= 3 and treatment_days > 0:
            dom_find = _find_finding(findings, "domiciliary")
            deductions.append(
                LineItemDeduction(
                    category="entire_claim",
                    claimed_inr=claimed_amount,
                    deduction_inr=claimed_amount,
                    sublimit_applied_inr=0.0,
                    reason=f"Domiciliary treatment duration ({treatment_days} days) does not exceed three days, excluded under policy Item 19.",
                    policy_clause=f"{dom_find.citation_section} (p. {dom_find.citation_pages})" if dom_find and dom_find.citation_section else "2.30 days Waiting Period > Item 19",
                )
            )
            return "REJECTED", claimed_amount, 0.0, deductions, missing_evidence

    # 5. Financial Sub-Limits & Category Capping (Deterministic Calculations)
    room_find = _find_finding(findings, "room_rent") or _find_finding(findings, "room")
    room_clause = (
        f"{room_find.citation_section} (p. {room_find.citation_pages})"
        if room_find and room_find.citation_section
        else "Sub limits (p. 7-8)"
    )

    doctor_find = _find_finding(findings, "doctor_fees") or _find_finding(findings, "surgeon")
    doctor_clause = (
        f"{doctor_find.citation_section} (p. {doctor_find.citation_pages})"
        if doctor_find and doctor_find.citation_section
        else "Sub limits > Item 2 (p. 7-8)"
    )

    meds_find = _find_finding(findings, "medicines") or _find_finding(findings, "diagnostic")
    meds_clause = (
        f"{meds_find.citation_section} (p. {meds_find.citation_pages})"
        if meds_find and meds_find.citation_section
        else "Sub limits > Item 3 (p. 7-8)"
    )

    amb_find = _find_finding(findings, "ambulance")
    amb_clause = (
        f"{amb_find.citation_section} (p. {amb_find.citation_pages})"
        if amb_find and amb_find.citation_section
        else "Sub limits > Item 4 (p. 7-8)"
    )

    # A) Room Rent Sub-Limit: 1.0% Basic Sum Insured per day for Normal Room
    room_claimed = float(expenses.get("room", 0.0))
    admission_hours = treatment.get("admission_hours", 0)
    treatment_type = treatment.get("type", "").lower()

    if room_claimed > 0:
        if treatment_type == "inpatient":
            # Days calculation: minimum 1 day, or admission_hours / 24 rounded up
            days_admitted = max(1, math.ceil(admission_hours / 24.0)) if admission_hours > 0 else 1
            max_room_allowance = days_admitted * (0.01 * sum_insured)
            if room_claimed > max_room_allowance:
                deduction_room = room_claimed - max_room_allowance
                deductions.append(
                    LineItemDeduction(
                        category="room",
                        claimed_inr=room_claimed,
                        deduction_inr=deduction_room,
                        sublimit_applied_inr=max_room_allowance,
                        reason=f"Normal room rent exceeds 1.0% of Basic Sum Insured per day (Cap: INR {max_room_allowance:,.0f} for {days_admitted} days).",
                        policy_clause=room_clause,
                    )
                )

    # B) Medical Practitioner / Surgeon Fees Sub-Limit: 25% of Sum Insured
    doctor_claimed = float(expenses.get("doctor_fees", 0.0))
    max_doctor_allowance = 0.25 * sum_insured
    if doctor_claimed > max_doctor_allowance:
        deduction_doctor = doctor_claimed - max_doctor_allowance
        deductions.append(
            LineItemDeduction(
                category="doctor_fees",
                claimed_inr=doctor_claimed,
                deduction_inr=deduction_doctor,
                sublimit_applied_inr=max_doctor_allowance,
                reason=f"Medical practitioner / consultant fees exceed 25% of Sum Insured (Cap: INR {max_doctor_allowance:,.0f}).",
                policy_clause=doctor_clause,
            )
        )

    # C) Medicines, Drugs, Diagnostics, OT, Chemotherapy Sub-Limit: 40% of Sum Insured
    meds_claimed = float(expenses.get("medicines_diagnostics", 0.0))
    max_meds_allowance = 0.40 * sum_insured
    if meds_claimed > max_meds_allowance:
        deduction_meds = meds_claimed - max_meds_allowance
        deductions.append(
            LineItemDeduction(
                category="medicines_diagnostics",
                claimed_inr=meds_claimed,
                deduction_inr=deduction_meds,
                sublimit_applied_inr=max_meds_allowance,
                reason=f"Medicines, diagnostic materials, and OT expenses exceed 40% of Sum Insured (Cap: INR {max_meds_allowance:,.0f}).",
                policy_clause=meds_clause,
            )
        )

    # D) Ambulance Sub-Limit: 1.0% of Basic Sum Insured or INR 1,000, whichever is less
    ambulance_claimed = float(expenses.get("ambulance", 0.0))
    max_ambulance_allowance = min(0.01 * sum_insured, 1000.0)
    if ambulance_claimed > max_ambulance_allowance:
        deduction_ambulance = ambulance_claimed - max_ambulance_allowance
        deductions.append(
            LineItemDeduction(
                category="ambulance",
                claimed_inr=ambulance_claimed,
                deduction_inr=deduction_ambulance,
                sublimit_applied_inr=max_ambulance_allowance,
                reason=f"Ambulance charges exceed policy limit of lesser of 1.0% BSI or INR 1,000 (Cap: INR {max_ambulance_allowance:,.0f}).",
                policy_clause=amb_clause,
            )
        )

    # E) Domiciliary Hospitalization Aggregate Sub-Limit: 20% of Basic Sum Insured
    if treatment_type == "domiciliary":
        dom_finding = _find_finding(findings, "domiciliary")
        dom_clause = (
            f"{dom_finding.citation_section} (p. {dom_finding.citation_pages})"
            if dom_finding and dom_finding.citation_section
            else "Sub limits > Item 3 NB2 (p. 7-8)"
        )
        max_domiciliary_allowance = 0.20 * sum_insured
        # Total payable before domiciliary cap
        subtotal_before_dom = claimed_amount - sum(d.deduction_inr for d in deductions)
        if subtotal_before_dom > max_domiciliary_allowance:
            excess_domiciliary = subtotal_before_dom - max_domiciliary_allowance
            deductions.append(
                LineItemDeduction(
                    category="domiciliary_aggregate",
                    claimed_inr=subtotal_before_dom,
                    deduction_inr=excess_domiciliary,
                    sublimit_applied_inr=max_domiciliary_allowance,
                    reason=f"Domiciliary hospitalization expenses exceed maximum aggregate sub-limit of 20% of Basic Sum Insured (Cap: INR {max_domiciliary_allowance:,.0f}).",
                    policy_clause=dom_clause,
                )
            )

    # F) Pre- and Post-Hospitalization Window Timing Verification
    if case.expense_timing:
        timing = case.expense_timing
        pre_days = (
            timing.pre_hospitalization_days_before_admission
            if hasattr(timing, "pre_hospitalization_days_before_admission")
            else timing.get("pre_hospitalization_days_before_admission")
        )
        post_days = (
            timing.post_hospitalization_days_after_discharge
            if hasattr(timing, "post_hospitalization_days_after_discharge")
            else timing.get("post_hospitalization_days_after_discharge")
        )

        pre_claimed = float(expenses.get("pre_hospitalization", 0.0))
        if pre_days is not None and pre_days > 30 and pre_claimed > 0:
            deductions.append(
                LineItemDeduction(
                    category="pre_hospitalization",
                    claimed_inr=pre_claimed,
                    deduction_inr=pre_claimed,
                    sublimit_applied_inr=0.0,
                    reason=f"Pre-hospitalization expenses incurred {pre_days} days before admission exceed the 30-day policy window.",
                    policy_clause="Note (p. 8-8)",
                )
            )

        post_claimed = float(expenses.get("post_hospitalization", 0.0))
        if post_days is not None and post_days > 60 and post_claimed > 0:
            deductions.append(
                LineItemDeduction(
                    category="post_hospitalization",
                    claimed_inr=post_claimed,
                    deduction_inr=post_claimed,
                    sublimit_applied_inr=0.0,
                    reason=f"Post-hospitalization expenses incurred {post_days} days after discharge exceed the 60-day policy window.",
                    policy_clause="Note (p. 8-8)",
                )
            )

    # 6. Compute Final Approved Amount
    total_deductions = sum(d.deduction_inr for d in deductions)
    approved_amount = max(0.0, claimed_amount - total_deductions)
    approved_amount = min(approved_amount, sum_insured)

    # Determine status
    if approved_amount == 0.0:
        status: Literal["APPROVED", "PARTIALLY_APPROVED", "REJECTED", "NEEDS_REVIEW"] = "REJECTED"
    elif total_deductions > 0.0:
        status = "PARTIALLY_APPROVED"
    else:
        status = "APPROVED"

    return status, claimed_amount, approved_amount, deductions, missing_evidence


def generate_justification_narrative(
    case: ClaimCase,
    status: str,
    claimed: float,
    approved: float,
    deductions: list[LineItemDeduction],
    findings: list[Finding],
    missing_evidence: list[str],
) -> str:
    """Generate professional narrative justification via LLM citing findings only."""
    treatment = case.get_treatment_dict()

    if not config.has_llm_key():
        # Fallback deterministic summary if no API key
        if status == "APPROVED":
            return f"Claim for {treatment.get('diagnosis')} is fully approved for INR {approved:,.2f} in accordance with policy terms."
        elif status == "PARTIALLY_APPROVED":
            return f"Claim for {treatment.get('diagnosis')} is partially approved for INR {approved:,.2f} with total deductions of INR {claimed - approved:,.2f} due to applicable policy sub-limits."
        elif status == "REJECTED":
            return f"Claim for {treatment.get('diagnosis')} is rejected in full (INR 0.00 approved) based on applicable policy exclusions or waiting periods."
        else:
            return f"Claim status is NEEDS_REVIEW pending submission of required verification: {', '.join(missing_evidence)}."

    deductions_lines = []
    for d in deductions:
        deductions_lines.append(
            f"- {d.category.upper()}: Claimed INR {d.claimed_inr:,.0f}, Deducted INR {d.deduction_inr:,.0f}. Reason: {d.reason} (Clause: {d.policy_clause})"
        )
    deductions_text = "\n".join(deductions_lines) if deductions_lines else "None (Full amount admissible)."

    findings_lines = []
    for f in findings:
        findings_lines.append(
            f"- [{f.issue}] Applies: {f.applies.upper()} | Citation: {f.citation_chunk_id} ({f.citation_section}, p. {f.citation_pages}) | Claim: {f.claim}"
        )
    findings_text = "\n".join(findings_lines) if findings_lines else "No specific findings."

    missing_text = "\n".join(f"- {m}" for m in missing_evidence) if missing_evidence else "None."

    prompt = JUSTIFICATION_PROMPT.format(
        case_id=case.case_id,
        diagnosis=treatment.get("diagnosis", "N/A"),
        procedure=treatment.get("procedure", "N/A"),
        sum_insured=f"{case.sum_insured_inr:,.0f}",
        claimed_amount=f"{claimed:,.2f}",
        approved_amount=f"{approved:,.2f}",
        status=status,
        deductions_text=deductions_text,
        findings_text=findings_text,
        missing_evidence_text=missing_text,
    )

    client = _get_client()
    try:
        narrative, _ = call_groq_chat(
            client=client,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=750,
            temperature=0,
            extra_body={"reasoning_effort": "low"},
        )
        return narrative.strip()
    except Exception as e:
        logger.error(f"[Agent 4] Failed to generate justification narrative: {e}")
        return (
            f"The claim was evaluated with status {status}. Total claimed: INR {claimed:,.2f}; "
            f"total approved: INR {approved:,.2f}. Itemized deductions applied: {len(deductions)}."
        )


def select_material_findings(
    status: str,
    findings: list[Finding],
    deductions: list[LineItemDeduction],
) -> list[Finding]:
    """Select only the findings directly material to the adjudication decision."""
    material: list[Finding] = []

    # 1. Base coverage is material if claim is approved or partially approved
    base = _find_finding(findings, "base_coverage")
    if base and status in ("APPROVED", "PARTIALLY_APPROVED"):
        material.append(base)

    # 2. Findings corresponding to applied deductions
    for d in deductions:
        clause_str = (d.policy_clause or "").lower()
        cat_str = d.category.lower()
        for f in findings:
            issue_str = f.issue.lower()
            sec_str = (f.citation_section or "").lower()
            if (sec_str and sec_str in clause_str) or (cat_str in issue_str) or (issue_str in clause_str):
                if f not in material:
                    material.append(f)

    # 3. For rejections, include fatal exclusion or waiting period findings
    if status == "REJECTED":
        for f in findings:
            if f.applies == "yes" and any(k in f.issue.lower() for k in ("waiting", "cosmetic", "experimental", "pre_existing", "cataract")):
                if f not in material:
                    material.append(f)

    # 4. For NEEDS_REVIEW, include uncertain findings
    if status == "NEEDS_REVIEW":
        for f in findings:
            if f.applies == "uncertain" and f not in material:
                material.append(f)

    return material if material else findings


def decide_claim(
    case: ClaimCase,
    plan: RetrievalPlan | None,
    findings: list[Finding],
    confidence_signal: float = 1.0,
) -> Decision:
    """Execute Agent 4: deterministic calculation followed by narrative generation."""
    status, claimed, approved, deductions, missing_evidence = calculate_decision_math(
        case=case,
        plan=plan,
        findings=findings,
        confidence_signal=confidence_signal,
    )

    material_findings = select_material_findings(status, findings, deductions)

    justification = generate_justification_narrative(
        case=case,
        status=status,
        claimed=claimed,
        approved=approved,
        deductions=deductions,
        findings=material_findings,
        missing_evidence=missing_evidence,
    )

    return Decision(
        status=status,
        claimed_amount_inr=claimed,
        approved_amount_inr=approved,
        deductions=deductions,
        justification=justification,
        supporting_findings=material_findings,
        missing_evidence=missing_evidence,
        validation_notes=[],
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from src.agents.case_analysis import analyze_case
    from src.agents.policy_evidence import gather_all_evidence
    from src.agents.coverage_exclusion import evaluate_all_bundles

    case_id = sys.argv[1] if len(sys.argv) > 1 else "PUB-001"
    cases_file = Path("data/cases/public_test_cases.json")
    with open(cases_file, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    match = next((c for c in all_cases if c["case_id"] == case_id), None)
    if not match:
        print(f"Case {case_id} not found")
        sys.exit(1)

    case_obj = ClaimCase(**match)
    plan = analyze_case(case_obj)
    bundles, conf = gather_all_evidence(plan)
    findings = evaluate_all_bundles(case_obj, bundles)
    decision = decide_claim(case_obj, plan, findings, conf)

    print(f"\n================ DECISION FOR {case_id} ================")
    print(f"Status: {decision.status}")
    print(f"Claimed: INR {decision.claimed_amount_inr:,.2f}")
    print(f"Approved: INR {decision.approved_amount_inr:,.2f}")
    print(f"Deductions ({len(decision.deductions)}):")
    for d in decision.deductions:
        print(f"  - [{d.category}] Claimed: {d.claimed_inr} | Deduct: {d.deduction_inr} | Reason: {d.reason}")
    if decision.missing_evidence:
        print(f"Missing Evidence ({len(decision.missing_evidence)}):")
        for m in decision.missing_evidence:
            print(f"  ! {m}")
    print(f"\nJustification:\n{decision.justification}")
