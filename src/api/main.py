"""FastAPI backend for the Aptino Multi-Agent Health Insurance Claim Decision Engine.

Endpoints:
  POST /adjudicate       - Submit a claim case for multi-agent adjudication
  POST /adjudicate/batch - Batch adjudication
  GET  /cases            - List all public test cases
  GET  /cases/{id}       - Get a specific test case by ID
  GET  /health           - Health check (retriever + index status)
  GET  /docs             - Swagger UI (built-in)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agents.orchestrator import get_retriever, run_case
from src.agents.state import CaseState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Aptino Claim Decision Engine",
    description=(
        "Multi-agent RAG system for health insurance claim adjudication. "
        "Reads the USGIC policy PDF and returns APPROVED / PARTIALLY_APPROVED / "
        "REJECTED / NEEDS_REVIEW with full citation trail."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("[API] Warming up retriever on startup...")
    try:
        get_retriever()
        logger.info("[API] Retriever ready.")
    except Exception as e:
        logger.error(f"[API] Retriever warmup failed: {e}")


class AdjudicateRequest(BaseModel):
    model_config = {"extra": "allow"}

    case_id: str = Field(..., description="Unique case identifier, e.g. PUB-001")
    policy_id: str
    policy_start_date: str
    claim_date: str
    sum_insured_inr: float
    continuous_coverage_months: int | float = 0
    prior_insurer_continuous_years: int | float = 0
    patient: dict[str, Any]
    hospital: dict[str, Any]
    treatment: dict[str, Any]
    expenses_inr: dict[str, float]
    documents: list[str] = Field(default_factory=list)
    task: str = ""
    evidence_context: dict[str, Any] | None = None
    expense_timing: dict[str, Any] | None = None
    prior_policy: dict[str, Any] | None = None


class AdjudicateResponse(BaseModel):
    case_id: str
    status: str
    claimed_amount_inr: float
    approved_amount_inr: float
    confidence_signal: float
    justification: str
    deductions: list[dict[str, Any]]
    missing_evidence: list[str]
    validation_notes: list[str]
    supporting_findings: list[dict[str, Any]]
    processing_time_seconds: float
    full_state: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    retriever_ready: bool
    chroma_collection_count: int | None
    bm25_index_ready: bool
    message: str


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    try:
        retriever = get_retriever()
        # Use the public retrieve() API to verify both dense + BM25 + reranker work
        test_chunks = retriever.retrieve("room rent sub limit hospitalization coverage")
        conf = retriever.confidence_signal(test_chunks)
        return HealthResponse(
            status="healthy",
            retriever_ready=True,
            chroma_collection_count=len(test_chunks),
            bm25_index_ready=True,
            message=f"Ready. Retrieved {len(test_chunks)} chunks on test query (conf={conf:.2f}).",
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            retriever_ready=False,
            chroma_collection_count=None,
            bm25_index_ready=False,
            message=f"Retriever error: {e}",
        )


@app.get("/cases", tags=["Test Cases"])
async def list_cases() -> list[dict[str, Any]]:
    cases_file = Path("data/cases/public_test_cases.json")
    if not cases_file.exists():
        raise HTTPException(status_code=404, detail="Test cases file not found.")
    with open(cases_file, "r", encoding="utf-8") as f:
        cases = json.load(f)
    summaries = []
    for c in cases:
        summaries.append({
            "case_id": c.get("case_id"),
            "policy_id": c.get("policy_id"),
            "claim_date": c.get("claim_date"),
            "sum_insured_inr": c.get("sum_insured_inr"),
            "diagnosis": c.get("treatment", {}).get("diagnosis"),
            "procedure": c.get("treatment", {}).get("procedure"),
            "treatment_type": c.get("treatment", {}).get("type"),
        })
    return summaries


@app.get("/cases/{case_id}", tags=["Test Cases"])
async def get_case(case_id: str) -> dict[str, Any]:
    cases_file = Path("data/cases/public_test_cases.json")
    if not cases_file.exists():
        raise HTTPException(status_code=404, detail="Test cases file not found.")
    with open(cases_file, "r", encoding="utf-8") as f:
        cases = json.load(f)
    match = next((c for c in cases if c.get("case_id") == case_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    return match


@app.post("/adjudicate", response_model=AdjudicateResponse, tags=["Adjudication"])
async def adjudicate(request: AdjudicateRequest) -> AdjudicateResponse:
    """Submit a claim case to the multi-agent adjudication pipeline."""
    raw = request.model_dump()
    logger.info(f"[API] Adjudicating case: {raw.get('case_id')}")

    t0 = time.perf_counter()
    try:
        state: CaseState = run_case(raw)
    except Exception as e:
        logger.error(f"[API] Pipeline error for {raw.get('case_id')}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline failed: {str(e)}",
        )
    elapsed = time.perf_counter() - t0

    dec = state.decision
    assert dec is not None

    return AdjudicateResponse(
        case_id=state.case.case_id,
        status=dec.status,
        claimed_amount_inr=dec.claimed_amount_inr,
        approved_amount_inr=dec.approved_amount_inr,
        confidence_signal=state.confidence_signal,
        justification=dec.justification,
        deductions=[d.model_dump() for d in dec.deductions],
        missing_evidence=dec.missing_evidence,
        validation_notes=dec.validation_notes,
        supporting_findings=[f.model_dump() for f in dec.supporting_findings],
        processing_time_seconds=round(elapsed, 2),
        full_state=state.model_dump(),
    )


@app.post("/adjudicate/batch", tags=["Adjudication"])
async def adjudicate_batch(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Batch adjudication endpoint."""
    results = []
    for raw in cases:
        case_id = raw.get("case_id", "unknown")
        t0 = time.perf_counter()
        try:
            state = run_case(raw)
            dec = state.decision
            assert dec is not None
            results.append({
                "case_id": case_id,
                "status": dec.status,
                "claimed_amount_inr": dec.claimed_amount_inr,
                "approved_amount_inr": dec.approved_amount_inr,
                "confidence_signal": state.confidence_signal,
                "processing_time_seconds": round(time.perf_counter() - t0, 2),
                "error": None,
            })
        except Exception as e:
            logger.error(f"[API] Batch error for {case_id}: {e}")
            results.append({
                "case_id": case_id,
                "status": "ERROR",
                "error": str(e),
                "processing_time_seconds": round(time.perf_counter() - t0, 2),
            })
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)

