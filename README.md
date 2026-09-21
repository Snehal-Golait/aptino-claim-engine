# Aptino AI Engineer Take-Home: Multi-Agent Health Insurance Claim Decision Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-orange.svg)](https://www.trychroma.com)
[![Groq](https://img.shields.io/badge/Groq-LPU%20Inference-f55036.svg)](https://groq.com)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)](https://python.org)

An end-to-end, production-grade **RAG + Multi-Agent Claim Decision Engine** that reads insurance policy documents (specifically the 17-page Universal Sompo Individual Health Insurance Policy), evaluates patient claims against policy terms, and outputs fully grounded, auditable decisions: `APPROVED`, `PARTIALLY_APPROVED`, `REJECTED`, or `NEEDS_REVIEW`.

Every factual assertion, deduction, and exclusion in every decision is **traceable to a specific section and page range of the policy PDF**, verified by a secondary **Natural Language Inference (NLI) Entailment Guard**.

---

## Live Public Deployment

The application is deployed and publicly reachable:
- **Streamlit Interactive UI**: [https://cool-otters-notice.loca.lt](https://cool-otters-notice.loca.lt)
  *(If prompted by localtunnel on first visit, enter tunnel password IP: `58.84.62.109`)*
- **FastAPI Backend Swagger**: `http://localhost:8000/docs` (local) / `/health` for readiness probe.

---

## Architecture Overview

```
                         ┌────────────────────────┐
                         │ Submitted Claim Case   │
                         │ (Pydantic ClaimCase)   │
                         └───────────┬────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │ Agent 1: Case Analysis & Retrieval Planner              │
        │ - Extracts diagnosis, procedure, timing, and billing    │
        │ - Generates targeted keyword & semantic queries         │
        └────────────────────────────┬────────────────────────────┘
                                     │ (Plan with N Queries)
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │ Agent 2: Policy Evidence Retriever (Per Query)          │
        │ ┌──────────────────────┐      ┌──────────────────────┐  │
        │ │ Dense Search (Chroma)│      │ Sparse Search (BM25) │  │
        │ └──────────┬───────────┘      └──────────┬───────────┘  │
        │            └────────────┬────────────────┘              │
        │                         ▼                               │
        │             Reciprocal Rank Fusion (RRF)                │
        │                         ▼                               │
        │            LLM Cross-Encoder Reranker                   │
        └────────────────────────────┬────────────────────────────┘
                                     │ (EvidenceBundles with Chunks)
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │ Agent 3: Coverage & Exclusion Evaluator                 │
        │ - One isolated evaluation per evidence bundle           │
        │ - Anti-Hallucination Guard: verifies chunk_id existence │
        └────────────────────────────┬────────────────────────────┘
                                     │ (Verified Findings)
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │ Agent 4: Deterministic Decision Engine                  │
        │ - Mathematical Accounting Ledger (Caps & Deductions)    │
        │ - LLM Narrative Justification Synthesis                 │
        └────────────────────────────┬────────────────────────────┘
                                     │ (Draft Decision)
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │ Agent 5: Validation & Entailment Guard                  │
        │ - NLI premise-hypothesis entailment check               │
        │ - Downgrades to NEEDS_REVIEW if ungrounded              │
        └────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
                         ┌────────────────────────┐
                         │ Final Adjudication     │
                         │ (Audited & Cited)      │
                         └────────────────────────┘
```

---

## Key Design Principles & Engineering Highlights

### 1. Document Structure-Aware Ingestion (`src/ingestion/chunker.py`)
- Standard character chunkers break legal and policy context across clauses.
- Our custom chunker uses font metadata (`pdfplumber`) to detect bold headings, numbered policy clauses (1 to 21), and definition boundaries (`"<Term> means..."`).
- Each chunk preserves `chunk_id`, `section` path (e.g. `DEFINITIONS > Hospital`), `page_start`, and `page_end`.

### 2. Hybrid Retrieval with RRF + LLM Reranking (`src/retrieval/`)
- **Dense Vector Search**: Chroma vector database using `all-MiniLM-L6-v2` embeddings for semantic intent.
- **Sparse Lexical Search**: Rank-BM25 over policy tokens for clause numbers and numerical thresholds (`"30 days"`, `"48 months"`, `"1.0%"`).
- **Reciprocal Rank Fusion (RRF)**: $RRF(d) = \sum_{m} \frac{1}{k + r_m(d)}$ with $k=60$.
- **LLM Reranking**: Scores top candidate chunks on a 0–10 relevance scale and provides a `confidence_signal`.

### 3. The 5-Agent Multi-Pipeline Architecture (`src/agents/`)
| Agent | Responsibility | Core Guardrail |
|---|---|---|
| **1. Case Analysis** | Parses claim, flags waiting periods & sub-limits | Explicit Pydantic `RetrievalPlan` schema |
| **2. Policy Evidence** | Retrieves and reranks policy chunks | Returns isolated `EvidenceBundle` per issue |
| **3. Coverage & Exclusion** | Evaluates applicability of policy terms | **Code Anti-Hallucination Guard**: Overrides finding to `uncertain` if cited `chunk_id` is hallucinated |
| **4. Decision Engine** | Computes deductible math and writes justification | **Deterministic Financial Math**: Python executes arithmetic; LLM only writes justification |
| **5. Validation Guard** | Entailment check against raw retrieved policy text | **NLI Entailment Gate**: Automatically places audit hold (`NEEDS_REVIEW`) if citation fails entailment |

---

## REST API Reference (`src/api/main.py`)

Run the API locally:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
Interactive Swagger documentation is available at `http://localhost:8000/docs`.

### Endpoints
- `GET  /health` — Verifies retriever readiness, Chroma collection count, and BM25 index status.
- `GET  /cases` — Returns the list of standard public test cases.
- `POST /analyze` — **(Assignment spec endpoint)** Analyzes a single claim and returns the specification-compliant response (`ADMISSIBLE` / `ADMISSIBLE_WITH_LIMITS` / `NOT_ADMISSIBLE` / `NEEDS_REVIEW`) with full citations, key findings, applicable limits, validation status, and agent trace.
- `POST /adjudicate` — Internal alias returning extended response (includes itemized deductions, full state, confidence signal).
- `POST /adjudicate/batch` — Batch adjudication for evaluation and bulk processing.

---

## Streamlit Frontend (`src/frontend/app.py`)

Run the interactive dashboard:
```bash
streamlit run src/frontend/app.py --server.port 8501
```

### Features:
1. **Interactive Case Runner**: Select from 12 public cases or 5 custom edge-case scenarios, or input custom JSON.
2. **Step-by-Step Pipeline Inspector**: View Agent 1 retrieval plan, Agent 2 evidence chunks with relevance scores, Agent 3 findings, and Agent 5 entailment logs.
3. **Financial Ledger**: Itemized breakdown of claimed vs. approved amounts with policy-cited deduction reasons.
4. **Policy Clause Viewer**: Inspect raw policy excerpts directly from the source PDF.

---

## Evaluation Suite (`eval/eval.py`)

Run the evaluation harness:
```bash
python -m eval.eval
```

The suite evaluates all **17 test cases** (12 public + 5 custom) across four key metrics:
1. **Decision Accuracy**: Percentage of cases where predicted status matches ground truth.
2. **Retrieval Recall@5**: Whether required policy clauses appear in the top-5 retrieved chunks.
3. **Citation Entailment Rate**: Percentage of citations verified by Agent 5 NLI guard.
4. **Failure Mode Deep Dive**: Root cause analysis and mitigation for edge cases.

Reports are generated at:
- `eval/results/eval_report.json`
- `eval/results/eval_report.md`

---

## 5 Custom Edge Cases (`data/cases/custom_cases/`)

1. **`CUST-001` — Portability Credit Insufficient**: Patient claims for cataract surgery with 3.5 months on current policy + 6 months prior credit (total 9.5 months), falling short of the mandatory 12-month waiting period (`REJECTED`).
2. **`CUST-002` — Dual Diagnosis with Cosmetic Split**: Inpatient bacterial pneumonia with incidental elective cosmetic scar revision; pneumonia is approved while cosmetic charges are deducted per Item 5 (`PARTIALLY_APPROVED`).
3. **`CUST-003` — Waiting Period Boundary (Day 35)**: Acute condition admitted on day 35 from inception; clears the initial 30-day waiting period and is fully reimbursed (`APPROVED`).
4. **`CUST-004` — Pre/Post Window Violation**: Pre-hospitalization at 45 days and post-hospitalization at 90 days exceed policy 30/60-day limits; disallowed while main admission is covered (`PARTIALLY_APPROVED`).
5. **`CUST-005` — Domiciliary Minimum 3-Day Rule**: Domiciliary treatment lasting only 2 days (48 hours); excluded by Item 19 of policy exclusions requiring > 3 days (`REJECTED`).

---

## Docker & Deployment

### Run via Docker Compose:
```bash
docker-compose up --build
```
- FastAPI: `http://localhost:8000` (docs at `/docs`)
- Streamlit: `http://localhost:8501`

### Environment Variables (`.env`):
```ini
GROQ_API_KEY=gsk_...
GROQ_MODEL=qwen/qwen3.8-27b
DENSE_TOP_K=15
SPARSE_TOP_K=15
FUSION_TOP_K=10
RERANK_TOP_K=5
```

---

## Running Unit & Integration Tests

```bash
pytest tests/ -v
```
The test suite covers:
- `tests/test_agents.py`: Claim math, waiting period deductions, sub-limit capping, and anti-hallucination guard.
- `tests/test_retrieval.py`: Dense search, BM25 sparse search, RRF rank fusion, and Retriever confidence signals.
- `tests/test_api.py`: FastAPI `/health`, `/cases`, and `/adjudicate` endpoints.

---

## Architecture Design Note

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full 2-page design document covering:
- Agent boundaries and structured state flow diagram
- Retrieval design: meaningful chunking, hybrid retrieval, RRF, LLM reranking
- Key design decisions and trade-offs table
- Abstention design (three distinct abstention mechanisms)
- Known limitations

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Snehal-Golait/aptino-claim-engine.git
cd aptino-claim-engine

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
copy .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Ingest the policy PDF (first-time only)
python -m src.ingestion.ingest

# 6. Run the API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 7. Run the frontend (new terminal)
streamlit run src/frontend/app.py --server.port 8501

# 8. Run evaluation
python -m eval.eval
```
