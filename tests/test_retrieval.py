"""Tests for hybrid retrieval: dense search, BM25 sparse search, RRF fusion, and Retriever facade."""
import pytest
from src.retrieval.dense import dense_search
from src.retrieval.sparse import sparse_search
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.retriever import Retriever


def test_sparse_bm25_search():
    """Verify BM25 index loads and returns relevant chunks for keyword queries."""
    results = sparse_search("cataract surgery waiting period", top_k=5)
    assert len(results) > 0
    # Top chunk should have chunk_id and non-zero BM25 score
    assert results[0].chunk_id != ""
    assert results[0].sparse_score is not None
    assert results[0].sparse_score > 0.0


def test_dense_chroma_search():
    """Verify Chroma collection loads and returns dense semantic matches."""
    results = dense_search("room rent capping and ICU charges", top_k=5)
    assert len(results) > 0
    assert results[0].chunk_id != ""
    assert results[0].dense_distance is not None


def test_rrf_fusion():
    """Verify reciprocal rank fusion merges and deduplicates dense + sparse results."""
    query = "domiciliary treatment definition hospital bed"
    d_results = dense_search(query, top_k=5)
    s_results = sparse_search(query, top_k=5)

    fused = reciprocal_rank_fusion(d_results, s_results, top_k=5)
    assert len(fused) <= 5
    assert len(fused) > 0
    # Every chunk should have fused_score > 0
    for c in fused:
        assert c.fused_score is not None
        assert c.fused_score > 0.0


def test_retriever_facade():
    """Verify Retriever returns chunks with confidence signal."""
    retriever = Retriever(top_k=5)
    results = retriever.retrieve("pre existing disease waiting period 48 months")
    assert len(results) > 0
    conf = retriever.confidence_signal(results)
    assert isinstance(conf, float)
    assert conf >= 0.0
