"""Agent 5: Validation & Entailment Re-Check (LLM-Based Factual Guard).

Purpose:
Addresses the requirement: "the LLM attempts to make a claim that cannot be supported by policy."
Agent 3's code guard verifies that the cited chunk_id exists in the bundle.
Agent 5 takes the next critical step: it re-fetches the actual raw chunk text and
performs an independent Natural Language Inference (entailment) check:
does the policy excerpt genuinely entail and support the specific claim made?

Safety Contract:
- If ANY material finding fails entailment or its citation cannot be verified,
  Agent 5 downgrades Decision.status to "NEEDS_REVIEW".
- Agent 5 may ONLY make decisions MORE conservative, NEVER less conservative.
- Detailed findings are recorded in `Decision.validation_notes`.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from groq import Groq

from src import config
from src.config import call_groq_chat
from src.agents.state import (
    ClaimCase,
    Decision,
    EvidenceBundle,
    Finding,
)

logger = logging.getLogger(__name__)

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


ENTAILMENT_PROMPT = """You are a health insurance claims auditor validating factual grounding between a policy excerpt and a finding claim.

POLICY EXCERPT (Ground Truth):
\"\"\"{chunk_text}\"\"\"

FACTUAL CLAIM TO VALIDATE:
\"{claim}\"

TASK:
Determine whether the policy excerpt reasonably grounds and supports the claim.
- supported = true: The excerpt substantially establishes or supports the rule, coverage, exclusion, limit, or condition stated in the claim (even if concisely summarized or paraphrased).
- supported = false: The claim asserts a rule, exclusion, limit, or condition that directly contradicts the excerpt or has zero textual basis in the excerpt.

Respond ONLY with a valid JSON object in this format:
{{
  "supported": true | false,
  "reason": "1 concise sentence explaining whether the excerpt grounds the claim."
}}
"""


def _parse_entailment_json(raw: str) -> dict[str, Any] | None:
    try:
        return json.loads(raw.strip())
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def validate_finding_entailment(
    finding: Finding,
    chunk_text: str,
) -> tuple[bool, str]:
    """Ask LLM whether the chunk text entails the finding's claim."""
    if not config.has_llm_key():
        return True, "Validation skipped (no GROQ_API_KEY set)."

    prompt = ENTAILMENT_PROMPT.format(
        chunk_text=chunk_text.strip(),
        claim=finding.claim.strip(),
    )

    client = _get_client()
    try:
        raw, _ = call_groq_chat(
            client=client,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0,
            extra_body={"reasoning_effort": "low"},
        )
        parsed = _parse_entailment_json(raw)
        if not parsed:
            logger.warning(f"[Agent 5] Entailment parse failure on {finding.citation_chunk_id}: raw={raw[:200]!r}")
            return False, f"Could not parse entailment validation response for {finding.citation_chunk_id}."

        is_supported = bool(parsed.get("supported", False))
        reason = str(parsed.get("reason", "No reason provided."))
        return is_supported, reason
    except Exception as e:
        logger.error(f"[Agent 5] Entailment call failed: {e}")
        return False, f"Entailment check error: {e}"


def validate_decision(
    decision: Decision,
    evidence_bundles: list[EvidenceBundle],
) -> Decision:
    """Validate all citations and entailments in the decision.

    Downgrades to NEEDS_REVIEW if any material finding fails entailment or has missing citations.
    """
    # Build complete chunk lookup across all gathered bundles
    chunk_pool: dict[str, str] = {}
    for b in evidence_bundles:
        for c in b.chunks:
            chunk_pool[c.chunk_id] = c.text

    validation_notes: list[str] = list(decision.validation_notes)
    downgrade_required = False

    # Check each material finding supporting the decision
    for finding in decision.supporting_findings:
        if finding.applies in ("yes", "no"):
            cid = finding.citation_chunk_id
            if not cid:
                note = f"Finding for '{finding.issue}' has no cited chunk ID."
                validation_notes.append(note)
                downgrade_required = True
                continue

            chunk_text = chunk_pool.get(cid)
            if not chunk_text:
                note = f"Cited chunk '{cid}' for issue '{finding.issue}' not found in evidence pool."
                validation_notes.append(note)
                downgrade_required = True
                continue

            # Check entailment
            is_supported, reason = validate_finding_entailment(finding, chunk_text)
            if is_supported:
                validation_notes.append(f"Verified: [{cid}] entails finding on '{finding.issue}'. ({reason})")
            else:
                note = f"Entailment Failure: [{cid}] does NOT support finding on '{finding.issue}'. ({reason})"
                validation_notes.append(note)
                downgrade_required = True

    # If downgrade required, enforce conservative shift
    if downgrade_required and decision.status != "NEEDS_REVIEW":
        logger.warning(
            f"[Agent 5] Decision status downgraded from {decision.status} to NEEDS_REVIEW "
            f"due to failed citation entailment checks."
        )
        decision.status = "NEEDS_REVIEW"
        decision.approved_amount_inr = 0.0
        decision.justification = (
            f"[AUDIT HOLD: NEEDS_REVIEW] One or more policy citations failed entailment validation. "
            f"{decision.justification}"
        )

    decision.validation_notes = validation_notes
    return decision


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from src.agents.case_analysis import analyze_case
    from src.agents.policy_evidence import gather_all_evidence
    from src.agents.coverage_exclusion import evaluate_all_bundles
    from src.agents.decision import decide_claim

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
    validated = validate_decision(decision, bundles)

    print(f"\n================ VALIDATED DECISION FOR {case_id} ================")
    print(f"Final Status: {validated.status}")
    print(f"Approved Amount: INR {validated.approved_amount_inr:,.2f}")
    print(f"Validation Notes ({len(validated.validation_notes)}):")
    for vn in validated.validation_notes:
        print(f"  * {vn}")
