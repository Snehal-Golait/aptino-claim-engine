"""Dense (semantic) retrieval against the Chroma collection built in ingest.py."""
from __future__ import annotations

from dataclasses import dataclass

import chromadb

from src import config


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    section: str
    page_start: int
    page_end: int
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_distance: float | None = None
    sparse_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None


_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        _collection = _client.get_collection(config.COLLECTION_NAME)
    return _collection


def dense_search(query: str, top_k: int = config.DENSE_TOP_K) -> list[RetrievedChunk]:
    """Return the top_k semantically closest chunks to `query`, ranked
    1..top_k (rank 1 = closest). Chroma returns cosine distance by default
    (lower = more similar) via its ONNX MiniLM embedding function."""
    collection = _get_collection()
    result = collection.query(query_texts=[query], n_results=top_k)

    chunks: list[RetrievedChunk] = []
    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]

    for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
        chunks.append(
            RetrievedChunk(
                chunk_id=cid,
                text=doc,
                section=meta["section"],
                page_start=meta["page_start"],
                page_end=meta["page_end"],
                dense_rank=rank,
                dense_distance=dist,
            )
        )
    return chunks


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "waiting period for pre-existing diseases"
    for c in dense_search(query, top_k=5):
        print(f"#{c.dense_rank} [{c.chunk_id}] {c.section} (p{c.page_start}-{c.page_end}) dist={c.dense_distance:.4f}")
        print("  ", c.text[:150].replace("\n", " "), "...")
