"""Centralized environment/config loading. Import this instead of scattering
os.getenv() calls across modules, so there's exactly one place that knows
the defaults and one place to change them."""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Prevent Windows console charmap encoding errors when printing PDF snippets
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# --- LLM ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
# Active, confirmed available models on Groq as of 2026:
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_MODELS = [
    "qwen/qwen3.8-27b",      # Fast, direct JSON output, no token-wasting reasoning overhead
    "openai/gpt-oss-120b",   # High-capacity fallback
    "openai/gpt-oss-20b",    # Fallback
]

# --- Paths ---
POLICY_PDF_PATH = os.getenv("POLICY_PDF_PATH", "data/policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf")
CHROMA_DIR = os.getenv("CHROMA_DIR", "storage/chroma")
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "storage/bm25/bm25_index.pkl")
COLLECTION_NAME = "policy_chunks"

# --- Retrieval tuning ---
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "15"))
SPARSE_TOP_K = int(os.getenv("SPARSE_TOP_K", "15"))
FUSION_TOP_K = int(os.getenv("FUSION_TOP_K", "10"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
RRF_K = 60  # standard RRF smoothing constant from the original paper

# --- Decision thresholds ---
MIN_RERANK_SCORE_FOR_CONFIDENT_DECISION = float(
    os.getenv("MIN_RERANK_SCORE_FOR_CONFIDENT_DECISION", "0.15")
)


def has_llm_key() -> bool:
    return bool(GROQ_API_KEY)


def call_groq_chat(
    client: Any,
    messages: list[dict[str, str]],
    max_tokens: int = 2048,
    temperature: float = 0.0,
    extra_body: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Execute chat completion with automatic model fallback on 429 rate limit."""
    last_err: Exception | None = None

    for model_name in GROQ_MODELS:
        try:
            # reasoning_effort is only valid for openai/gpt-oss reasoning models
            call_extra = extra_body
            current_max_tokens = max_tokens
            if "gpt-oss" in model_name:
                if call_extra is None:
                    call_extra = {"reasoning_effort": "low"}
            elif "qwen" in model_name:
                call_extra = None
                # Groq free-tier enforces 1,000 output tokens per minute (OTPM) on Qwen.
                # Clamping to 800 prevents preemptive 429 rejection on request creation.
                current_max_tokens = min(max_tokens, 800)
            else:
                call_extra = None

            kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": current_max_tokens,
            }
            if call_extra:
                kwargs["extra_body"] = call_extra

            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "stop"
            return content, finish_reason
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            if any(term in err_str for term in ["429", "rate limit", "tokens per day", "tpd", "resource_exhausted"]):
                print(f"[Groq LLM] Model {model_name} rate-limited ({e}). Falling back to next model.")
                continue
            raise e

    raise last_err if last_err is not None else RuntimeError("All Groq models failed.")
