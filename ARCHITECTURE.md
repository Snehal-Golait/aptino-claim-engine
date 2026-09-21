# Architecture & Design Note
## Aptino Multi-Agent Health Insurance Claim Decision Engine

**System**: Policy-aware RAG + multi-agent claim adjudication engine  
**Policy**: Universal Sompo Individual Health Insurance — UNIHLIP18004V011718 (17 pages)  
**Stack**: Python 3.11, FastAPI, ChromaDB, Rank-BM25, Groq LLM (free tier)

---

## 1. Agent Boundaries & Structured State Flow

The system uses five genuinely specialized agents that communicate exclusively via a typed Pydantic state object (`CaseState`). No agent shares a prompt context with another; each receives only its required input slice.

```
ClaimCase (raw JSON)
       │
       ▼ CaseState.case
┌──────────────────────────────────────────────────────────┐
│ Agent 1: Case Analysis & Retrieval Planner               │
│ INPUT : CaseState.case                                   │
│ OUTPUT: CaseState.plan → RetrievalPlan(queries=[...])    │
│ ROLE  : Produces targeted, clause-specific retrieval     │
│         queries (NOT one monolithic "find coverage"      │
│         query). Flags waiting period, day-care, PED,     │
│         domiciliary, portability, sub-limit issues.      │
└──────────────────────────┬───────────────────────────────┘
                           │ RetrievalPlan
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Agent 2: Policy Evidence Retriever                       │
│ INPUT : One RetrievalQuery per plan item                 │
│ OUTPUT: CaseState.evidence_bundles → [EvidenceBundle]    │
│ ROLE  : Executes dense + sparse + RRF + LLM reranking    │
│         independently per query issue. Each bundle is    │
│         self-contained with chunk metadata.              │
└──────────────────────────┬───────────────────────────────┘
                           │ [EvidenceBundle]
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Agent 3: Coverage & Exclusion Evaluator                  │
│ INPUT : One EvidenceBundle at a time                     │
│ OUTPUT: CaseState.findings → [Finding]                   │
│ ROLE  : Isolated per-issue LLM judgment. Applies         │
│         anti-hallucination guard: if the LLM cites a    │
│         chunk_id NOT in the bundle, the finding is       │
│         overridden to applies=uncertain.                 │
└──────────────────────────┬───────────────────────────────┘
                           │ [Finding] (verified)
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Agent 4: Deterministic Decision Engine                   │
│ INPUT : CaseState.case + CaseState.findings              │
│ OUTPUT: CaseState.decision → Decision (draft)            │
│ ROLE  : Python code performs all financial arithmetic    │
│         (room rent 1% BSI cap, ICU 2% cap, ambulance     │
│         min(1% BSI, INR 1000), 30/60-day pre/post        │
│         hospitalization windows, waiting periods).       │
│         LLM only writes the cited narrative justification│
│         from the verified finding set.                   │
└──────────────────────────┬───────────────────────────────┘
                           │ Decision (draft)
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Agent 5: Validation & NLI Entailment Guard               │
│ INPUT : Decision.supporting_findings + raw chunk text    │
│ OUTPUT: CaseState.decision (final, with audit notes)     │
│ ROLE  : For each material finding, tests entailment:     │
│         "Does the raw policy chunk TEXT actually support  │
│         the cited claim?" If not, sets NEEDS_REVIEW and  │
│         zeroes approved_amount. Provides PASS/FAIL audit │
│         trail for every citation.                        │
└──────────────────────────────────────────────────────────┘
```

**State contract**: Every agent reads from and writes to `CaseState`. No agent can see another agent's internal prompt or chain-of-thought. The exposed `/analyze` trace contains only agent names, action summaries, retrieval counts, and validation status — never hidden chain-of-thought.

---

## 2. Retrieval Design

### 2.1 Meaningful Chunking (not fixed-size)
`src/ingestion/chunker.py` uses a three-pass strategy on `pdfplumber` font metadata:

1. **Pass 1 — Heading detection**: Bold or large-font runs mark new section boundaries. Headings like "WHAT WE COVER", "GENERAL EXCLUSIONS", "DEFINITIONS" each start a new chunk.
2. **Pass 2 — Numbered clause detection**: Lines matching `^\d+\.` (policy clauses 1–21) force a chunk split regardless of proximity to the prior heading.
3. **Pass 3 — Definition boundary detection**: Runs matching `"<Term> means..."` are split into standalone definition chunks.

Each chunk preserves: `chunk_id` (deterministic slug), `section` (e.g. `DEFINITIONS > Hospital`), `page_start`, `page_end`, and raw `text`.

### 2.2 Hybrid Retrieval Pipeline
| Stage | Method | Purpose |
|---|---|---|
| Dense | ChromaDB + `all-MiniLM-L6-v2` | Semantic intent matching |
| Sparse | Rank-BM25 over policy tokens | Exact keyword hits (clause numbers, percentages, durations) |
| Fusion | Reciprocal Rank Fusion (k=60) | Combine ranked lists without score normalization |
| Reranking | LLM 0-10 relevance scoring | Fine-grained relevance with confidence signal |

**Why RRF over score normalization?** BM25 and dense scores are on incompatible scales. RRF avoids any rescaling assumption by working purely on rank positions — `RRF(d) = Σ 1 / (k + r_m(d))` where `r_m(d)` is the rank of document `d` in method `m`.

### 2.3 Per-Issue Retrieval Isolation
Agent 2 runs the full retrieval pipeline independently for each planned query. This prevents a high-scoring "base coverage" chunk from drowning out a critical "30-day waiting period" chunk when competing in a single joint retrieval. It also makes citation accountability straightforward: each finding maps to exactly one query and one evidence bundle.

---

## 3. Key Design Decisions & Trade-offs

| Decision | Rationale | Trade-off |
|---|---|---|
| **Deterministic financial math in Python** | LLMs make arithmetic errors under token pressure; sub-limit calculations must be exact | Agent 4 is harder to generalize to novel policy structures not anticipated at design time |
| **Anti-hallucination chunk guard (Agent 3)** | LLMs occasionally cite plausible-but-fabricated `chunk_id`s; code verification is free | May be overly conservative if the LLM paraphrases a real chunk ID |
| **NLI entailment gate (Agent 5)** | Secondary entailment LLM call verifies that cited text actually supports the decision claim | Increases latency and token usage; TPD exhaustion can truncate late calls on long eval runs |
| **Groq free tier with two-model fallback** | Zero cost; 200K TPD split across `qwen/qwen3.8-27b` (primary) and `openai/gpt-oss-120b` (fallback) | Daily token limits constrain batch eval throughput; not suitable for high-volume production |
| **ChromaDB with bundled ONNX embeddings** | No separate embedding server; no torch dependency; deployment image stays ~300 MB | Single-node only; not suitable for large-scale horizontal scaling |
| **No torch dependency** | Keeps Docker image small for free-tier hosting (Render, HF Spaces) | Cannot use BGE cross-encoder reranker; LLM-based reranking used instead |
| **`extra="allow"` on all Pydantic models** | Assignment explicitly requires tolerating extra/unknown JSON fields gracefully | Slightly reduces type strictness; unknown fields silently pass through |

---

## 4. Abstention Design

The system abstains (`NEEDS_REVIEW`) in three distinct ways, each traceable:

1. **Missing evidence fields (Agent 1)**: Nulls in `hospital_registered`, `medical_necessity_confirmed`, or `prior_policy.database_and_claim_history_received` are flagged in the retrieval plan. Agent 4 escalates automatically to `NEEDS_REVIEW` with an explicit `missing_evidence` list.
2. **Uncertain Agent 3 findings**: If the LLM cannot conclude `applies=yes|no` on a fatal dimension (waiting period, exclusion), it returns `applies=uncertain`. Agent 4 treats any `uncertain` finding on a blocking dimension as an abstention trigger.
3. **Failed NLI entailment (Agent 5)**: If a material citation fails the premise-hypothesis entailment check, the decision is downgraded to `NEEDS_REVIEW`, the `approved_amount` is zeroed, and the failing claim is appended to `validation.unsupported_claims`.

---

## 5. Known Limitations

- **Groq TPD cap (200K tokens/day)**: Long evaluation runs hit the daily limit. Two eval cases (PUB-005, PUB-008) were affected during the evaluation run when the daily cap was exhausted. Mitigation: `time.sleep(1.5)` in the eval loop; two-model fallback chain.
- **Localtunnel deployment**: The public URL is ephemeral and requires the local machine to stay running. For a permanent deployment, Render or Hugging Face Spaces is recommended.
- **Recall on unanticipated sub-clauses**: Agent 1 must proactively generate a query for every applicable issue. Cases with unusual clause combinations not covered by the planning heuristics (e.g., domiciliary treatment + Item 19 minimum-3-day rule) can miss retrieval. This is the root cause of the CUST-005 failure.
- **Single-document corpus**: The current architecture is designed for one policy document. Multi-insurer or multi-policy support would require corpus-level retrieval routing.
