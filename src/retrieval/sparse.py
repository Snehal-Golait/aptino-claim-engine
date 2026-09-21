"""Sparse (lexical) retrieval using the BM25 index built in ingest.py.

Dense embeddings are great at "what concept is this about" but can miss
exact terms that matter a lot in insurance policy language — a specific
clause number, "24 hours", "48 months", a named disease. BM25 catches those
exact-term matches that a semantic model can blur past, which is exactly
why the assignment asks for hybrid retrieval rather than dense-only.
"""
from __future__ import annotations

import pickle

from src import config
from src.ingestion.ingest import tokenize
from src.retrieval.dense import RetrievedChunk

_index_cache: dict | None = None


def _get_index() -> dict:
    global _index_cache
    if _index_cache is None:
        with open(config.BM25_INDEX_PATH, "rb") as f:
            _index_cache = pickle.load(f)
    return _index_cache


def sparse_search(query: str, top_k: int = config.SPARSE_TOP_K) -> list[RetrievedChunk]:
    idx = _get_index()
    bm25 = idx["bm25"]
    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    chunks: list[RetrievedChunk] = []
    for rank, i in enumerate(ranked, start=1):
        if scores[i] <= 0:
            continue  # no lexical overlap at all — not a real match
        meta = idx["metadatas"][i]
        chunks.append(
            RetrievedChunk(
                chunk_id=idx["chunk_ids"][i],
                text=idx["documents"][i],
                section=meta["section"],
                page_start=meta["page_start"],
                page_end=meta["page_end"],
                sparse_rank=rank,
                sparse_score=float(scores[i]),
            )
        )
    return chunks


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "waiting period for pre-existing diseases"
    for c in sparse_search(query, top_k=5):
        print(f"#{c.sparse_rank} [{c.chunk_id}] {c.section} (p{c.page_start}-{c.page_end}) score={c.sparse_score:.3f}")
        print("  ", c.text[:150].replace("\n", " "), "...")
