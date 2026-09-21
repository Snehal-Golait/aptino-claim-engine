"""Agent 2: Policy Evidence Gatherer (Deterministic, No LLM Call).

Takes the RetrievalPlan produced by Agent 1 and executes each query against
the Phase 2 Retriever (hybrid dense Chroma + sparse BM25 + RRF + rerank).
Packages results into typed EvidenceBundle objects, computes retrieval confidence,
and prepares the grounded factual pool.
"""
from __future__ import annotations

import json
from typing import Sequence

from src.agents.state import (
    ClaimCase,
    EvidenceBundle,
    EvidenceChunk,
    RetrievalPlan,
    RetrievalQuery,
)
from src.retrieval.dense import RetrievedChunk
from src.retrieval.retriever import Retriever


def _to_evidence_chunk(rc: RetrievedChunk) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=rc.chunk_id,
        section=rc.section,
        page_start=rc.page_start,
        page_end=rc.page_end,
        text=rc.text,
        dense_rank=rc.dense_rank,
        sparse_rank=rc.sparse_rank,
        fused_score=rc.fused_score,
        rerank_score=rc.rerank_score,
    )


def gather_evidence_for_query(
    query_item: RetrievalQuery,
    retriever: Retriever,
) -> EvidenceBundle:
    raw_chunks = retriever.retrieve(query_item.query)
    evidence_chunks = [_to_evidence_chunk(c) for c in raw_chunks]
    max_conf = retriever.confidence_signal(raw_chunks)

    return EvidenceBundle(
        issue=query_item.issue,
        query=query_item.query,
        rationale=query_item.rationale,
        chunks=evidence_chunks,
        max_confidence=max_conf,
    )


def gather_all_evidence(
    plan: RetrievalPlan,
    retriever: Retriever | None = None,
) -> tuple[list[EvidenceBundle], float]:
    """Execute all plan queries and return the list of bundles plus the overall confidence signal."""
    if retriever is None:
        retriever = Retriever()

    bundles: list[EvidenceBundle] = []
    max_scores: list[float] = []

    for q in plan.queries:
        bundle = gather_evidence_for_query(q, retriever)
        bundles.append(bundle)
        max_scores.append(bundle.max_confidence)

    overall_confidence = max(max_scores) if max_scores else 0.0
    return bundles, overall_confidence


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from src.agents.case_analysis import analyze_case

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
    retriever = Retriever()
    bundles, conf = gather_all_evidence(plan, retriever)

    print(f"Evidence gathered for {case_id}:")
    print(f"Overall confidence signal: {conf}")
    print(f"Bundles retrieved: {len(bundles)}")
    for b in bundles:
        print(f"\n--- Issue: {b.issue} (max rerank: {b.max_confidence}) ---")
        for c in b.chunks[:3]:
            print(f"  [{c.chunk_id}] {c.section} (p{c.page_start}-{c.page_end})")
            print(f"    {c.text[:120].replace(chr(10), ' ')}...")
