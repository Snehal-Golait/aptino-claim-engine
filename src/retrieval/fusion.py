"""
Reciprocal Rank Fusion (RRF) — merges the dense and sparse ranked lists into
one ranking without needing to normalize two incomparable score scales
(cosine distance vs BM25 score). For each chunk:

    fused_score = sum( 1 / (RRF_K + rank_in_list) )  over every list it appears in

A chunk that ranks well in BOTH lists rises to the top; a chunk that only
one retriever found still gets a chance, weighted by how high it ranked
there. This is the standard formulation from Cormack et al. 2009.
"""
from __future__ import annotations

from src import config
from src.retrieval.dense import RetrievedChunk, dense_search
from src.retrieval.sparse import sparse_search


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    top_k: int = config.FUSION_TOP_K,
    k: int = config.RRF_K,
) -> list[RetrievedChunk]:
    merged: dict[str, RetrievedChunk] = {}

    for chunk in dense_results:
        merged[chunk.chunk_id] = chunk

    for chunk in sparse_results:
        if chunk.chunk_id in merged:
            merged[chunk.chunk_id].sparse_rank = chunk.sparse_rank
            merged[chunk.chunk_id].sparse_score = chunk.sparse_score
        else:
            merged[chunk.chunk_id] = chunk

    for chunk in merged.values():
        score = 0.0
        if chunk.dense_rank is not None:
            score += 1.0 / (k + chunk.dense_rank)
        if chunk.sparse_rank is not None:
            score += 1.0 / (k + chunk.sparse_rank)
        chunk.fused_score = score

    ranked = sorted(merged.values(), key=lambda c: c.fused_score, reverse=True)
    return ranked[:top_k]


def hybrid_search(
    query: str,
    dense_top_k: int = config.DENSE_TOP_K,
    sparse_top_k: int = config.SPARSE_TOP_K,
    fusion_top_k: int = config.FUSION_TOP_K,
) -> list[RetrievedChunk]:
    dense_results = dense_search(query, top_k=dense_top_k)
    sparse_results = sparse_search(query, top_k=sparse_top_k)
    return reciprocal_rank_fusion(dense_results, sparse_results, top_k=fusion_top_k)


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "waiting period for pre-existing diseases"
    for c in hybrid_search(query):
        print(
            f"fused={c.fused_score:.4f}  dense_rank={c.dense_rank}  sparse_rank={c.sparse_rank}  "
            f"[{c.chunk_id}] {c.section} (p{c.page_start}-{c.page_end})"
        )
        print("  ", c.text[:150].replace("\n", " "), "...")
