"""Agent 1: Case Analysis (Deterministic, Rule-Based, No LLM Call).

Turns a raw ClaimCase into a targeted RetrievalPlan.
Because the case is already structured JSON, generating query requirements
(e.g., checking day-care if admission_hours < 24, checking PED waiting period if
pre_existing=True) is a deterministic mapping. This guarantees zero hallucination,
sub-millisecond execution, zero API cost, and 100% test reproducibility.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from src.agents.state import ClaimCase, RetrievalPlan, RetrievalQuery

# Keyword topics to trigger specialized clause queries
TOPIC_KEYWORDS: dict[str, tuple[str, str]] = {
    "cataract": (
        "cataract eye surgery waiting period first year sub limit",
        "Clause covering Cataract surgery waiting period and day care procedure",
    ),
    "cancer": (
        "cancer critical illness chemotherapy radiotherapy sub limits",
        "Sub-limits and coverage rules for cancer treatment and oncology",
    ),
    "cosmetic": (
        "cosmetic aesthetic surgery plastic surgery exclusion",
        "Exclusion clause for cosmetic or aesthetic treatments",
    ),
    "thyroid": (
        "pre existing thyroid disorder waiting period 48 months",
        "Waiting period provisions for pre-existing thyroid conditions",
    ),
    "hernia": (
        "hernia hydrocele waiting period first year",
        "Waiting period for hernia and related conditions",
    ),
    "piles": (
        "fistula piles waiting period first year",
        "Waiting period for piles, fistula, and colorectal conditions",
    ),
    "dental": (
        "dental treatment surgery exclusion",
        "Exclusions regarding dental treatment",
    ),
    "maternity": (
        "maternity pregnancy childbirth exclusion",
        "Exclusion regarding pregnancy, childbirth, and maternity",
    ),
}


def _calculate_days_between(start_date_str: str, claim_date_str: str) -> int:
    try:
        d1 = datetime.strptime(start_date_str, "%Y-%m-%d")
        d2 = datetime.strptime(claim_date_str, "%Y-%m-%d")
        return (d2 - d1).days
    except Exception:
        return 999


def analyze_case(case: ClaimCase) -> RetrievalPlan:
    """Analyze a ClaimCase and produce a deterministic RetrievalPlan with flags."""
    queries: list[RetrievalQuery] = []
    flags: dict[str, Any] = {}

    treatment = case.get_treatment_dict()
    expenses = case.get_expenses_dict()
    diag_proc = f"{treatment.get('diagnosis', '')} {treatment.get('procedure', '')}".lower()

    # 1. Evidence context check (unresolved nulls / missing essential evidence)
    unresolved_fields: list[str] = []
    if case.evidence_context:
        for k, v in case.evidence_context.items():
            if v is None:
                unresolved_fields.append(k)
            elif v is False and k in ("hospital_minimum_criteria_documented", "hospital_registered"):
                unresolved_fields.append(f"{k}_not_met")

    flags["has_unresolved_evidence_fields"] = len(unresolved_fields) > 0
    flags["unresolved_fields"] = unresolved_fields

    # 2. Base Hospitalization / Diagnosis Coverage
    diagnosis = treatment.get("diagnosis", "Hospitalization")
    procedure = treatment.get("procedure", "Medical treatment")
    queries.append(
        RetrievalQuery(
            issue="base_coverage",
            query=f"hospitalization expenses what we cover {diagnosis} {procedure}",
            rationale="Verify base admissibility and covered expenses for the admitted condition.",
        )
    )

    # 3. Admission Hours & Day-Care Treatment (< 24 hours for hospital admissions)
    admission_hours = treatment.get("admission_hours", 0)
    treatment_type = treatment.get("type", "").lower()
    if treatment_type != "domiciliary" and (admission_hours < 24 or treatment_type == "day_care"):
        queries.append(
            RetrievalQuery(
                issue="day_care_admission",
                query="day care treatment 24 hours hospitalization specified procedures annexure",
                rationale="Evaluate whether procedure qualifies for waiver of 24-hour hospitalization requirement.",
            )
        )
        flags["is_day_care_or_short_stay"] = True

    # 4. Domiciliary Hospitalization
    if treatment_type == "domiciliary":
        queries.append(
            RetrievalQuery(
                issue="domiciliary_treatment",
                query="domiciliary hospitalization sub limit 20 percent basic sum insured conditions three days",
                rationale="Assess domiciliary treatment conditions, 3-day threshold, and 20% sum insured sub-limit.",
            )
        )
        flags["is_domiciliary"] = True

    # 5. Pre-Existing Disease (PED) Waiting Period
    if treatment.get("pre_existing", False):
        queries.append(
            RetrievalQuery(
                issue="pre_existing_disease",
                query="pre-existing diseases 48 months continuous coverage waiting period exclusion",
                rationale="Check applicability of 48-month waiting period for declared pre-existing condition.",
            )
        )
        flags["has_pre_existing_condition"] = True

    # 6. Initial 30-Day Waiting Period
    elapsed_days = _calculate_days_between(case.policy_start_date, case.claim_date)
    coverage_months = case.continuous_coverage_months
    flags["elapsed_days_since_inception"] = elapsed_days
    if coverage_months < 1 or elapsed_days < 30:
        queries.append(
            RetrievalQuery(
                issue="initial_waiting_period",
                query="30 days waiting period first thirty days from inception illness exclusion accident",
                rationale="Assess if claim falls within statutory 30-day initial waiting period from inception.",
            )
        )
        flags["in_initial_waiting_period_window"] = True

    # 7. Portability & Continuity Credit
    if case.prior_insurer_continuous_years > 0 or case.prior_policy:
        queries.append(
            RetrievalQuery(
                issue="portability_credit",
                query="portability continuous coverage waiting period reduction database claim history",
                rationale="Determine portability continuity credits for waiting period reduction.",
            )
        )
        flags["has_portability_claim"] = True

    # 8. Experimental / Unproven Treatment Exclusion
    if treatment.get("experimental", False) or "experimental" in diag_proc:
        queries.append(
            RetrievalQuery(
                issue="experimental_treatment",
                query="unproven experimental treatment exclusion",
                rationale="Verify exclusion of experimental or unproven therapies under general exclusions.",
            )
        )
        flags["is_experimental"] = True

    # 9. Specialized Topic Clauses (Cataract, Cancer, Cosmetic, etc.)
    for topic, (q_text, rationale) in TOPIC_KEYWORDS.items():
        if topic in diag_proc:
            queries.append(
                RetrievalQuery(
                    issue=f"topic_{topic}",
                    query=q_text,
                    rationale=rationale,
                )
            )
            flags[f"has_topic_{topic}"] = True

    # 10. Pre- and Post-Hospitalization Window & Limits
    pre_exp = expenses.get("pre_hospitalization", 0)
    post_exp = expenses.get("post_hospitalization", 0)
    if pre_exp > 0 or post_exp > 0 or case.expense_timing:
        queries.append(
            RetrievalQuery(
                issue="pre_post_hospitalization",
                query="pre-hospitalization 30 days post-hospitalization 60 days note expenses limit",
                rationale="Verify pre-hospitalization (30 days) and post-hospitalization (60 days) reimbursability.",
            )
        )
        flags["has_pre_post_expenses"] = True

    # 11. Hospital Definition / Registration Requirements
    is_network = (
        case.hospital.network_provider
        if hasattr(case.hospital, "network_provider")
        else (case.hospital.get("network_provider") if isinstance(case.hospital, dict) else None)
    )
    if flags.get("has_unresolved_evidence_fields") or is_network is False:
        queries.append(
            RetrievalQuery(
                issue="hospital_definition",
                query="hospital means definition registration 10 in-patient beds qualified nursing staff OT",
                rationale="Verify whether treating institution fulfills policy criteria for Hospital/Nursing Home.",
            )
        )

    # 12. Granular Financial Sub-Limits: Room, Ambulance, Doctor Fees, Diagnostics
    if expenses.get("room", 0) > 0:
        queries.append(
            RetrievalQuery(
                issue="room_rent_sub_limit",
                query="normal room expenses 1.0 percent Basic Sum Insured intensive care 2 percent",
                rationale="Retrieve daily sub-limit for normal room rent and ICU.",
            )
        )
    if expenses.get("ambulance", 0) > 0:
        queries.append(
            RetrievalQuery(
                issue="ambulance_sub_limit",
                query="ambulance charges 1.0 percent Basic Sum Insured Rupees 1000 whichever is less",
                rationale="Retrieve sub-limit on ambulance charges per claim.",
            )
        )
    if expenses.get("doctor_fees", 0) > 0.20 * case.sum_insured_inr:
        queries.append(
            RetrievalQuery(
                issue="doctor_fees_sub_limit",
                query="medical practitioner consultant surgeon fees limit 25 percent Sum Assured",
                rationale="Retrieve professional and surgeon fee cap of 25%.",
            )
        )
    if expenses.get("medicines_diagnostics", 0) > 0.35 * case.sum_insured_inr:
        queries.append(
            RetrievalQuery(
                issue="medicines_diagnostics_sub_limit",
                query="medicines drugs diagnostic materials operation theatre chemotherapy limit 40 percent Sum Insured",
                rationale="Retrieve medicines, diagnostics, and OT sub-limit of 40%.",
            )
        )

    return RetrievalPlan(
        case_id=case.case_id,
        queries=queries,
        flags=flags,
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    case_id = sys.argv[1] if len(sys.argv) > 1 else "PUB-001"
    cases_file = Path("data/cases/public_test_cases.json")
    with open(cases_file, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    match = next((c for c in all_cases if c["case_id"] == case_id), None)
    if not match:
        print(f"Case {case_id} not found in {cases_file}")
        sys.exit(1)

    case_obj = ClaimCase(**match)
    plan = analyze_case(case_obj)
    print(f"Plan for {plan.case_id}:")
    print(f"Flags: {json.dumps(plan.flags, indent=2)}")
    print(f"Generated {len(plan.queries)} queries:")
    for i, q in enumerate(plan.queries, 1):
        print(f"  {i}. [{q.issue}] {q.query}")
        print(f"     Rationale: {q.rationale}")
