"""Single entry point the agents will call: query in, ranked+cited chunks out."""
from __future__ import annotations

from dataclasses import asdict

from src import config
from src.retrieval.dense import RetrievedChunk
from src.retrieval.reranker import hybrid_search_and_rerank


class Retriever:
    def __init__(self, top_k: int = config.RERANK_TOP_K):
        self.top_k = top_k

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return hybrid_search_and_rerank(query, top_k=self.top_k)

    def retrieve_as_dicts(self, query: str) -> list[dict]:
        """Convenience for agents that want plain dicts (e.g. to embed in a
        prompt or a Pydantic model) instead of RetrievedChunk objects."""
        return [asdict(c) for c in self.retrieve(query)]

    def confidence_signal(self, chunks: list[RetrievedChunk]) -> float:
        """A single 0-10-ish number the Decision/Validation agents can use
        as one input to the abstain-vs-decide call. Highest rerank score
        among the returned chunks; 0 if nothing came back at all."""
        if not chunks:
            return 0.0
        scored = [c.rerank_score for c in chunks if c.rerank_score is not None]
        return max(scored) if scored else 0.0


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "cataract surgery waiting period portability"
    retriever = Retriever()
    results = retriever.retrieve(query)
    print(f"Query: {query!r}\nConfidence signal: {retriever.confidence_signal(results)}\n")
    for c in results:
        print(f"[{c.chunk_id}] {c.section} (p{c.page_start}-{c.page_end}) rerank={c.rerank_score}")
        print("  ", c.text[:200].replace("\n", " "), "...")
        print()
