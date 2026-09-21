"""Integration tests for FastAPI endpoints (/health, /cases, /adjudicate)."""
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify /health returns 200 OK and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["retriever_ready"] is True


def test_list_cases_endpoint():
    """Verify /cases returns list of 12 public cases."""
    response = client.get("/cases")
    assert response.status_code == 200
    cases = response.json()
    assert len(cases) == 12
    assert any(c["case_id"] == "PUB-001" for c in cases)


def test_adjudicate_endpoint():
    """Verify /adjudicate endpoint processes a case and returns complete decision schema."""
    cases_file = Path("data/cases/public_test_cases.json")
    with open(cases_file, "r", encoding="utf-8") as f:
        cases = json.load(f)
    case_payload = cases[0]  # PUB-001

    response = client.post("/adjudicate", json=case_payload)
    assert response.status_code == 200
    res = response.json()
    assert res["case_id"] == "PUB-001"
    assert res["status"] in ("REJECTED", "NEEDS_REVIEW", "PARTIALLY_APPROVED", "APPROVED")
    assert "justification" in res
    assert "processing_time_seconds" in res
    assert res["claimed_amount_inr"] == 163200.0
