"""Agent 3: Coverage and Exclusion Evaluator (LLM-based with Code Anti-Hallucination Guard).

Evaluates each evidence bundle independently (one targeted call per bundle,
never one giant unstructured prompt for the whole case).

Critical Anti-Hallucination Guard:
After parsing the LLM's structured JSON response, code strictly verifies that
the cited `chunk_id` actually exists in the bundle's retrieved chunks. If the
LLM cites a non-existent, hallucinated, or malformed chunk ID, code immediately
overrides the finding to `applies="uncertain"` and nulls the citation.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from groq import Groq

from src import config
from src.agents.state import ClaimCase, EvidenceBundle, Finding
from src.config import call_groq_chat

logger = logging.getLogger(__name__)

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


EVAL_PROMPT = """You are a health insurance policy adjudicator. Evaluate whether the policy excerpts support coverage, an exclusion, a waiting period, or a sub-limit restriction for this claim.

CLAIM CONTEXT:
- Case ID: {case_id}
- Diagnosis: {diagnosis}
- Procedure: {procedure}
- Treatment Type: {treatment_type}
- Admission Hours: {admission_hours}
- Pre-Existing Condition: {pre_existing}
- Continuous Coverage Months: {coverage_months}
- Prior Insurer Continuous Years: {prior_years}
- Experimental Therapy: {experimental}
- Sum Insured INR: {sum_insured}

ISSUE UNDER EVALUATION:
- Issue: {issue}
- Evaluation Goal: {rationale}

RETRIEVED POLICY EXCERPTS:
{excerpts}

RULES:
1. Ground your decision ONLY in the provided policy excerpts. Do NOT use outside legal or insurance knowledge.
2. Decide 'applies':
   - 'yes': The clause/restriction/coverage affirmatively applies to the claim based on the facts and excerpt.
   - 'no': The clause/restriction does not apply (e.g. condition cleared or exception satisfied).
   - 'uncertain': The excerpts do not provide enough specific evidence to confirm whether this applies or not.
3. 'citation_chunk_id': MUST be the exact chunk ID (e.g. "sub-limits-00") from the excerpts above that directly supports your finding.
4. 'claim': A single concise factual sentence stating strictly what the cited excerpt explicitly says, using the excerpt's exact conditions and terms without extrapolating.
5. 'reasoning': 2-3 sentences explaining how the policy text maps to the case facts.

Respond ONLY with a valid JSON object in this format:
{{
  "applies": "yes" | "no" | "uncertain",
  "citation_chunk_id": "EXACT_CHUNK_ID_FROM_EXCERPTS",
  "claim": "Clear factual statement grounded strictly in cited excerpt",
  "reasoning": "Reasoning mapping excerpt text to claim facts"
}}
"""


def _format_bundle_excerpts(bundle: EvidenceBundle) -> str:
    lines = []
    for c in bundle.chunks:
        snippet = c.text.strip().replace("\n", " ")
        lines.append(f"--- Chunk ID: {c.chunk_id} | Section: {c.section} (Pages {c.page_start}-{c.page_end}) ---\n{snippet}")
    return "\n\n".join(lines)


def _parse_llm_json(raw: str) -> dict[str, Any] | None:
    # Attempt direct load
    try:
        return json.loads(raw.strip())
    except Exception:
        pass

    # Extract outermost JSON object via regex
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return None


def evaluate_bundle(
    case: ClaimCase,
    bundle: EvidenceBundle,
) -> Finding:
    """Evaluate an evidence bundle using Groq LLM with strict code citation verification."""
    treatment = case.get_treatment_dict()
    valid_chunks = {c.chunk_id: c for c in bundle.chunks}

    if not config.has_llm_key():
        # Fallback if no LLM key configured
        top_chunk = bundle.chunks[0] if bundle.chunks else None
        return Finding(
            issue=bundle.issue,
            applies="uncertain",
            citation_chunk_id=top_chunk.chunk_id if top_chunk else None,
            citation_section=top_chunk.section if top_chunk else None,
            citation_pages=f"{top_chunk.page_start}-{top_chunk.page_end}" if top_chunk else None,
            claim=f"Unverified finding for {bundle.issue} (no GROQ_API_KEY configured).",
            reasoning="LLM evaluation skipped because GROQ_API_KEY is not set.",
        )

    prompt = EVAL_PROMPT.format(
        case_id=case.case_id,
        diagnosis=treatment.get("diagnosis", "N/A"),
        procedure=treatment.get("procedure", "N/A"),
        treatment_type=treatment.get("type", "N/A"),
        admission_hours=treatment.get("admission_hours", 0),
        pre_existing=treatment.get("pre_existing", False),
        coverage_months=case.continuous_coverage_months,
        prior_years=case.prior_insurer_continuous_years,
        experimental=treatment.get("experimental", False),
        sum_insured=case.sum_insured_inr,
        issue=bundle.issue,
        rationale=bundle.rationale,
        excerpts=_format_bundle_excerpts(bundle),
    )

    client = _get_client()
    try:
        raw, finish_reason = call_groq_chat(
            client=client,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0,
            extra_body={"reasoning_effort": "low"},
        )
    except Exception as e:
        logger.error(f"[Agent 3] Groq API call failed for issue {bundle.issue}: {e}")
        return Finding(
            issue=bundle.issue,
            applies="uncertain",
            citation_chunk_id=None,
            citation_section=None,
            citation_pages=None,
            claim=f"API error during evaluation of {bundle.issue}",
            reasoning=f"LLM call raised exception: {e}",
        )

    parsed = _parse_llm_json(raw)
    if not parsed:
        logger.warning(
            f"[Agent 3] Could not parse JSON for issue {bundle.issue}. "
            f"finish_reason={finish_reason!r}, raw snippet={raw[:300]!r}"
        )
        return Finding(
            issue=bundle.issue,
            applies="uncertain",
            citation_chunk_id=None,
            citation_section=None,
            citation_pages=None,
            claim=f"Unable to parse LLM evaluation for {bundle.issue}",
            reasoning=f"Parse failure with finish_reason={finish_reason}",
        )

    raw_applies = str(parsed.get("applies", "uncertain")).lower().strip()
    if raw_applies not in ("yes", "no", "uncertain"):
        raw_applies = "uncertain"

    cited_id = parsed.get("citation_chunk_id")
    claim_text = parsed.get("claim", "")
    reasoning_text = parsed.get("reasoning", "")

    # === CRITICAL ANTI-HALLUCINATION GUARD ===
    # Verify the cited chunk_id was actually in the retrieved bundle
    if cited_id and cited_id in valid_chunks:
        cited_chunk = valid_chunks[cited_id]
        citation_section = cited_chunk.section
        citation_pages = f"{cited_chunk.page_start}-{cited_chunk.page_end}"
        final_applies = raw_applies
    else:
        # Hallucinated or malformed chunk ID! Force uncertain and null citation
        logger.warning(
            f"[Agent 3 Anti-Hallucination Guard] Cited chunk '{cited_id}' NOT in retrieved bundle for issue '{bundle.issue}'. "
            f"Overriding applies to 'uncertain'."
        )
        final_applies = "uncertain"
        cited_id = None
        citation_section = None
        citation_pages = None
        reasoning_text = f"[Guard Triggered: Hallucinated citation '{parsed.get('citation_chunk_id')}'] {reasoning_text}"

    return Finding(
        issue=bundle.issue,
        applies=final_applies,  # type: ignore
        citation_chunk_id=cited_id,
        citation_section=citation_section,
        citation_pages=citation_pages,
        claim=claim_text,
        reasoning=reasoning_text,
    )


def evaluate_all_bundles(
    case: ClaimCase,
    bundles: list[EvidenceBundle],
) -> list[Finding]:
    """Evaluate all bundles, returning the complete list of vetted findings."""
    return [evaluate_bundle(case, b) for b in bundles]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from src.agents.case_analysis import analyze_case
    from src.agents.policy_evidence import gather_all_evidence

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

    print(f"\nFindings for {case_id}:")
    for f in findings:
        print(f"\n[Issue: {f.issue}] Applies: {f.applies.upper()}")
        print(f"  Citation: {f.citation_chunk_id} | {f.citation_section} (Pages: {f.citation_pages})")
        print(f"  Claim: {f.claim}")
        print(f"  Reasoning: {f.reasoning}")
