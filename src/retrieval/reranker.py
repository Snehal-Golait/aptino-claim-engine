"""
Reranking stage: takes the RRF-fused candidates and re-orders them by how
well they actually answer the query.

Design decision: we use an LLM-based listwise reranker (via Groq's free
tier) instead of a local cross-encoder (e.g. BGE/ms-marco). We deliberately
kept `torch`/`sentence-transformers` out of this project's dependencies —
a cross-encoder needs them, and on a free-tier deployment box (Render/HF
Spaces free instances) that's a multi-hundred-MB dependency for a single
model call per request. An LLM call we're already paying for latency-wise
(the agents need one anyway) does double duty here. This is a documented
trade-off, not an oversight — see README "Design Decisions".

If no GROQ_API_KEY is configured yet, reranking is skipped and the RRF
fusion order is kept as-is (with a printed warning), so retrieval is still
testable before you wire up the LLM.
"""
from __future__ import annotations

import json
import re

from groq import Groq

from src import config
from src.retrieval.dense import RetrievedChunk
from src.retrieval.fusion import hybrid_search

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


_RERANK_PROMPT = """You are scoring how well each policy excerpt helps answer a query about a health insurance claim.

Query: {query}

Excerpts:
{excerpts}

For each excerpt, give a relevance score from 0 (irrelevant) to 10 (directly answers the query).
Respond ONLY with a JSON array of numbers in the same order as the excerpts, e.g. [7, 2, 9, 0, 5].
No other text."""


def _format_excerpts(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for i, c in enumerate(chunks):
        snippet = c.text[:500].replace("\n", " ")
        lines.append(f"[{i}] ({c.section}) {snippet}")
    return "\n\n".join(lines)


def _parse_scores(raw: str, expected_len: int) -> list[float] | None:
    match = re.search(r"\[[\d.,\s]+\]", raw)
    if not match:
        return None
    try:
        scores = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(scores, list) or not scores:
        return None
    parsed = [float(s) for s in scores if isinstance(s, (int, float))]
    if len(parsed) < expected_len:
        parsed.extend([0.0] * (expected_len - len(parsed)))
    elif len(parsed) > expected_len:
        parsed = parsed[:expected_len]
    return parsed


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int = config.RERANK_TOP_K) -> list[RetrievedChunk]:
    if not chunks:
        return []

    if not config.has_llm_key():
        print("[reranker] WARNING: GROQ_API_KEY not set — skipping LLM rerank, keeping RRF fusion order.")
        for c in chunks:
            c.rerank_score = c.fused_score
        return chunks[:top_k]

    prompt = _RERANK_PROMPT.format(query=query, excerpts=_format_excerpts(chunks))
    client = _get_client()
    try:
        raw, finish_reason = config.call_groq_chat(
            client=client,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=600,
            extra_body={"reasoning_effort": "low"},
        )
    except Exception as e:
        print(f"[reranker] WARNING: LLM rerank call failed ({e}) — falling back to fusion order.")
        for c in chunks:
            c.rerank_score = c.fused_score
        return chunks[:top_k]

    scores = _parse_scores(raw, len(chunks))

    if scores is None:
        print(
            f"[reranker] WARNING: could not parse LLM rerank output "
            f"(finish_reason={finish_reason!r}, raw={raw[:300]!r}) "
            "-- falling back to fusion order."
        )
        for c in chunks:
            c.rerank_score = c.fused_score
        return chunks[:top_k]

    for c, s in zip(chunks, scores):
        c.rerank_score = s

    ranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
    return ranked[:top_k]


def hybrid_search_and_rerank(query: str, top_k: int = config.RERANK_TOP_K) -> list[RetrievedChunk]:
    fused = hybrid_search(query)
    return rerank(query, fused, top_k=top_k)


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "waiting period for pre-existing diseases"
    for c in hybrid_search_and_rerank(query):
        print(f"rerank={c.rerank_score}  fused={c.fused_score:.4f}  [{c.chunk_id}] {c.section} (p{c.page_start}-{c.page_end})")
        print("  ", c.text[:150].replace("\n", " "), "...")
