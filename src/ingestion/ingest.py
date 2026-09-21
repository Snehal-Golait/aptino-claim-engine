"""
Build the two retrieval indexes from the chunked policy:

  1. Chroma collection (dense/semantic) — uses chromadb's bundled ONNX
     all-MiniLM-L6-v2 embedding function. No torch/GPU required, which
     keeps the footprint small for free-tier deployment (Render/HF Spaces
     free instances are CPU-only and disk/RAM constrained).
  2. A pickled BM25Okapi index (sparse/lexical) over the same chunks.

Run:  python -m src.ingestion.ingest
"""
from __future__ import annotations

import os
import pickle
import re

import chromadb
from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk, chunk_policy_pdf

POLICY_PDF_PATH = os.getenv("POLICY_PDF_PATH", "data/policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf")
CHROMA_DIR = os.getenv("CHROMA_DIR", "storage/chroma")
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "storage/bm25/bm25_index.pkl")
COLLECTION_NAME = "policy_chunks"

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Simple, dependency-free tokenizer shared by index build and query time."""
    return _TOKEN_PATTERN.findall(text.lower())


def build_dense_index(chunks: list[Chunk]) -> None:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "section": c.section,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "source": os.path.basename(POLICY_PDF_PATH),
            }
            for c in chunks
        ],
    )
    print(f"Dense index: {collection.count()} chunks in Chroma at {CHROMA_DIR}")


def build_sparse_index(chunks: list[Chunk]) -> None:
    os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
    tokenized_corpus = [tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    payload = {
        "bm25": bm25,
        "chunk_ids": [c.chunk_id for c in chunks],
        "documents": [c.text for c in chunks],
        "metadatas": [
            {
                "section": c.section,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "source": os.path.basename(POLICY_PDF_PATH),
            }
            for c in chunks
        ],
    }
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"Sparse index: {len(chunks)} chunks pickled at {BM25_INDEX_PATH}")


def main() -> None:
    print(f"Chunking {POLICY_PDF_PATH} ...")
    chunks = chunk_policy_pdf(POLICY_PDF_PATH)
    print(f"Produced {len(chunks)} chunks.\n")

    build_dense_index(chunks)
    build_sparse_index(chunks)
    print("\nIngestion complete.")


if __name__ == "__main__":
    main()
