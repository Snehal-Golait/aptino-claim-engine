"""Streamlit Frontend for the Aptino Multi-Agent Health Insurance Claim Decision Engine.

Features:
- Live adjudication using the local pipeline (no API round-trip required)
- Load any of the 12 public test cases from the bundled JSON
- Display rich decision output: status badge, approved amount, itemized deductions,
  policy citations with page numbers, confidence signal, validation notes
- Full agent trace view (collapsible)
- "Run All Cases" eval summary table
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import logging

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Aptino Claim Engine",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap");

html, body, [class*="css"] { font-family: "Inter", sans-serif; }

.status-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}
.badge-approved      { background: #d1fae5; color: #065f46; }
.badge-partial       { background: #fef3c7; color: #92400e; }
.badge-rejected      { background: #fee2e2; color: #991b1b; }
.badge-needs-review  { background: #dbeafe; color: #1e40af; }
.badge-error         { background: #f3f4f6; color: #374151; }

.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 12px;
    padding: 20px;
    color: white;
    text-align: center;
}
.metric-card .label { font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px; }
.metric-card .value { font-size: 1.8rem; font-weight: 700; }

.citation-box {
    background: #f8fafc;
    border-left: 4px solid #6366f1;
    border-radius: 4px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.85rem;
}
.deduction-row {
    background: #fff7ed;
    border-left: 4px solid #f97316;
    border-radius: 4px;
    padding: 10px 14px;
    margin-bottom: 8px;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load test cases
# ---------------------------------------------------------------------------
CASES_FILE = Path("data/cases/public_test_cases.json")

@st.cache_data
def load_cases() -> list[dict]:
    if not CASES_FILE.exists():
        return []
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


ALL_CASES = load_cases()
CASE_MAP = {c["case_id"]: c for c in ALL_CASES}

# ---------------------------------------------------------------------------
# Pipeline runner (cached per-case by hash of the JSON)
# ---------------------------------------------------------------------------
def run_pipeline(raw_case: dict) -> dict:
    """Run the multi-agent pipeline and return serializable result dict."""
    from src.agents.orchestrator import run_case
    t0 = time.perf_counter()
    state = run_case(raw_case)
    elapsed = round(time.perf_counter() - t0, 2)
    dec = state.decision
    assert dec is not None
    return {
        "case_id": state.case.case_id,
        "status": dec.status,
        "claimed_amount_inr": dec.claimed_amount_inr,
        "approved_amount_inr": dec.approved_amount_inr,
        "confidence_signal": state.confidence_signal,
        "justification": dec.justification,
        "deductions": [d.model_dump() for d in dec.deductions],
        "missing_evidence": dec.missing_evidence,
        "validation_notes": dec.validation_notes,
        "supporting_findings": [f.model_dump() for f in dec.supporting_findings],
        "processing_time_seconds": elapsed,
        "full_state": state.model_dump(),
    }


# ---------------------------------------------------------------------------
# Status badge helper
# ---------------------------------------------------------------------------
STATUS_BADGE = {
    "APPROVED": ("badge-approved", "✅ APPROVED"),
    "PARTIALLY_APPROVED": ("badge-partial", "⚠️ PARTIALLY APPROVED"),
    "REJECTED": ("badge-rejected", "❌ REJECTED"),
    "NEEDS_REVIEW": ("badge-needs-review", "🔍 NEEDS REVIEW"),
}

def render_status_badge(status: str) -> None:
    css, label = STATUS_BADGE.get(status, ("badge-error", f"❓ {status}"))
    st.markdown(f'<div class="status-badge {css}">{label}</div>', unsafe_allow_html=True)

def fmt_inr(amount: float) -> str:
    return f"₹{amount:,.2f}"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/medical-doctor.png", width=64)
    st.title("🏥 Aptino\nClaim Engine")
    st.caption("Multi-Agent RAG · USGIC Policy")
    st.divider()

    mode = st.radio("Mode", ["📋 Adjudicate Case", "📊 Run All Cases", "📖 About"], index=0)
    st.divider()

    if mode == "📋 Adjudicate Case":
        st.subheader("Select Case")
        case_options = ["(Custom JSON)"] + [c["case_id"] for c in ALL_CASES]
        selected_case_id = st.selectbox(
            "Test Case",
            options=case_options,
            help="Select a pre-loaded test case or paste custom JSON below",
        )
        st.divider()
        st.caption("💡 The pipeline runs 5 agents:\n1. Case Analysis\n2. Policy Evidence\n3. Coverage & Exclusion\n4. Decision Math\n5. Validation")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

# ── ABOUT ──────────────────────────────────────────────────────────────────
if mode == "📖 About":
    st.title("Aptino Claim Decision Engine")
    st.markdown("""
**A RAG + Multi-Agent system for health insurance claim adjudication.**

Built against the USGIC CSC Individual Health Insurance 2017-2018 policy PDF.

## Architecture
| Agent | Role | LLM? |
|-------|------|-------|
| 1 · Case Analysis | Deterministic query planning & flag extraction | ❌ No |
| 2 · Policy Evidence | Hybrid BM25 + Dense retrieval → RRF fusion → CrossEncoder rerank | ❌ No |
| 3 · Coverage & Exclusion | LLM evaluates each retrieved bundle independently | ✅ Yes |
| 4 · Decision | Deterministic Python math + LLM narrative justification | ✅ Yes |
| 5 · Validation | LLM entailment re-check; downgrades to NEEDS_REVIEW on failure | ✅ Yes |

## Retrieval Pipeline
- **Dense**: ChromaDB + `all-MiniLM-L6-v2` embeddings
- **Sparse**: BM25Okapi (rank_bm25)
- **Fusion**: Reciprocal Rank Fusion (RRF, k=60)
- **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Decision Labels
- **APPROVED** – Full claim approved within policy limits  
- **PARTIALLY_APPROVED** – Claim approved with sub-limit deductions  
- **REJECTED** – Excluded by waiting period or explicit exclusion clause  
- **NEEDS_REVIEW** – Insufficient evidence / entailment failure / abstention signal
    """)

# ── RUN ALL CASES ──────────────────────────────────────────────────────────
elif mode == "📊 Run All Cases":
    st.title("📊 Batch Evaluation — All Public Cases")
    st.info("This runs the full multi-agent pipeline for all 12 public test cases sequentially. Expect ~3-8 minutes total.", icon="ℹ️")

    if st.button("▶️ Run All Cases Now", type="primary", use_container_width=True):
        results = []
        prog_bar = st.progress(0.0, text="Starting pipeline...")
        status_container = st.empty()

        for i, case in enumerate(ALL_CASES):
            cid = case["case_id"]
            status_container.info(f"Processing {cid} ({i+1}/{len(ALL_CASES)})...")
            try:
                r = run_pipeline(case)
                results.append(r)
            except Exception as e:
                results.append({
                    "case_id": cid,
                    "status": "ERROR",
                    "claimed_amount_inr": 0.0,
                    "approved_amount_inr": 0.0,
                    "confidence_signal": 0.0,
                    "processing_time_seconds": 0.0,
                    "error": str(e),
                })
            prog_bar.progress((i + 1) / len(ALL_CASES), text=f"Completed {cid}")

        status_container.success(f"✅ All {len(ALL_CASES)} cases processed!")
        st.session_state["batch_results"] = results

    if "batch_results" in st.session_state:
        results = st.session_state["batch_results"]
        st.divider()
        st.subheader("Results Summary")

        import pandas as pd
        rows = []
        for r in results:
            rows.append({
                "Case ID": r["case_id"],
                "Status": r.get("status", "ERROR"),
                "Claimed (₹)": f"{r.get('claimed_amount_inr', 0):,.0f}",
                "Approved (₹)": f"{r.get('approved_amount_inr', 0):,.0f}",
                "Confidence": f"{r.get('confidence_signal', 0):.2f}",
                "Time (s)": r.get("processing_time_seconds", 0),
                "Error": r.get("error", ""),
            })
        df = pd.DataFrame(rows)

        def color_status(val):
            colors = {
                "APPROVED": "background-color: #d1fae5",
                "PARTIALLY_APPROVED": "background-color: #fef3c7",
                "REJECTED": "background-color: #fee2e2",
                "NEEDS_REVIEW": "background-color: #dbeafe",
                "ERROR": "background-color: #f3f4f6",
            }
            return colors.get(val, "")

        st.dataframe(df.style.applymap(color_status, subset=["Status"]), use_container_width=True)

        # Stats
        col1, col2, col3, col4 = st.columns(4)
        statuses = [r.get("status") for r in results]
        col1.metric("✅ Approved", statuses.count("APPROVED"))
        col2.metric("⚠️ Partial", statuses.count("PARTIALLY_APPROVED"))
        col3.metric("❌ Rejected", statuses.count("REJECTED"))
        col4.metric("🔍 Review", statuses.count("NEEDS_REVIEW"))

# ── ADJUDICATE CASE ────────────────────────────────────────────────────────
else:
    st.title("🏥 Claim Adjudication")

    # Case input
    if selected_case_id == "(Custom JSON)":
        st.subheader("Custom Claim Input")
        custom_json_str = st.text_area(
            "Paste claim JSON",
            height=300,
            placeholder='{"case_id": "CUSTOM-001", "policy_id": "...", ...}',
        )
        raw_case = None
        if custom_json_str.strip():
            try:
                raw_case = json.loads(custom_json_str)
                st.success("✅ Valid JSON parsed")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
    else:
        raw_case = CASE_MAP.get(selected_case_id)
        if raw_case:
            st.subheader(f"Case: {selected_case_id}")
            treatment = raw_case.get("treatment", {})
            expenses = raw_case.get("expenses_inr", {})

            col1, col2, col3 = st.columns(3)
            col1.markdown(f"""
**Diagnosis:** {treatment.get("diagnosis", "N/A")}  
**Procedure:** {treatment.get("procedure", "N/A")}  
**Type:** {treatment.get("type", "N/A").upper()}  
**Hours:** {treatment.get("admission_hours", 0)}h
""")
            col2.markdown(f"""
**Hospital:** {raw_case.get("hospital", {}).get("name", "N/A")}  
**Network:** {"✅ Yes" if raw_case.get("hospital", {}).get("network_provider") else "❌ No"}  
**Sum Insured:** {fmt_inr(raw_case.get("sum_insured_inr", 0))}  
**Coverage:** {raw_case.get("continuous_coverage_months", 0)} months
""")
            total_claimed = sum(float(v) for v in expenses.values())
            col3.markdown(f"""
**Total Claimed:** {fmt_inr(total_claimed)}  
**Room:** {fmt_inr(expenses.get("room", 0))}  
**Doctor Fees:** {fmt_inr(expenses.get("doctor_fees", 0))}  
**Medicines:** {fmt_inr(expenses.get("medicines_diagnostics", 0))}  
**Ambulance:** {fmt_inr(expenses.get("ambulance", 0))}
""")

            with st.expander("📄 Full Case JSON"):
                st.json(raw_case)

    # Run pipeline
    st.divider()
    run_btn = st.button(
        "🚀 Run Adjudication Pipeline",
        disabled=(raw_case is None),
        type="primary",
        use_container_width=True,
    )

    if run_btn and raw_case:
        with st.spinner("🤖 Running 5-agent pipeline... (30–90 seconds)"):
            try:
                result = run_pipeline(raw_case)
                st.session_state["last_result"] = result
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

    # Display result
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]

        # Only show if same case
        if result.get("case_id") != (raw_case or {}).get("case_id"):
            if not (selected_case_id == "(Custom JSON)"):
                st.session_state.pop("last_result", None)
                st.stop()

        st.divider()
        st.subheader("🏆 Adjudication Decision")

        render_status_badge(result["status"])

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-card">
<div class="label">CLAIMED</div>
<div class="value">{fmt_inr(result["claimed_amount_inr"])}</div>
</div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
<div class="label">APPROVED</div>
<div class="value">{fmt_inr(result["approved_amount_inr"])}</div>
</div>""", unsafe_allow_html=True)
        with col3:
            total_deductions = result["claimed_amount_inr"] - result["approved_amount_inr"]
            st.markdown(f"""<div class="metric-card">
<div class="label">TOTAL DEDUCTIONS</div>
<div class="value">{fmt_inr(total_deductions)}</div>
</div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-card">
<div class="label">CONFIDENCE</div>
<div class="value">{result["confidence_signal"]:.2f}</div>
</div>""", unsafe_allow_html=True)

        st.caption(f"⏱️ Processing time: {result['processing_time_seconds']}s")

        # Justification
        st.divider()
        st.subheader("📝 Decision Justification")
        st.info(result["justification"])

        # Deductions
        if result["deductions"]:
            st.divider()
            st.subheader(f"💸 Itemized Deductions ({len(result['deductions'])})")
            for d in result["deductions"]:
                st.markdown(f"""<div class="deduction-row">
<strong>{d["category"].upper().replace("_", " ")}</strong><br>
Claimed: {fmt_inr(d["claimed_inr"])} → Deducted: <strong>{fmt_inr(d["deduction_inr"])}</strong>
{f" (Cap: {fmt_inr(d['sublimit_applied_inr'])})" if d.get("sublimit_applied_inr") is not None else ""}<br>
<em>Reason:</em> {d["reason"]}<br>
<em>Policy Clause:</em> <code>{d.get("policy_clause", "N/A")}</code>
</div>""", unsafe_allow_html=True)

        # Citations
        if result["supporting_findings"]:
            st.divider()
            st.subheader(f"📚 Policy Citations ({len(result['supporting_findings'])})")
            for f in result["supporting_findings"]:
                applies_icon = {"yes": "✅", "no": "❎", "uncertain": "❓"}.get(f["applies"], "❓")
                st.markdown(f"""<div class="citation-box">
<strong>{applies_icon} {f["issue"].replace("_", " ").title()}</strong> · applies: <code>{f["applies"].upper()}</code><br>
<em>Section:</em> {f.get("citation_section", "N/A")} · <em>Pages:</em> {f.get("citation_pages", "N/A")} · <em>Chunk:</em> <code>{f.get("citation_chunk_id", "N/A")}</code><br>
{f["claim"]}<br>
<small>💬 {f["reasoning"]}</small>
</div>""", unsafe_allow_html=True)

        # Missing evidence
        if result["missing_evidence"]:
            st.divider()
            st.subheader("⚠️ Missing Evidence / Abstention Signals")
            for m in result["missing_evidence"]:
                st.warning(m)

        # Validation notes
        if result["validation_notes"]:
            st.divider()
            st.subheader("🛡️ Agent 5 Validation Notes")
            for vn in result["validation_notes"]:
                if "fail" in vn.lower() or "downgrade" in vn.lower():
                    st.error(vn)
                else:
                    st.success(vn)

        # Full trace
        with st.expander("🔬 Full Agent Trace (JSON)"):
            st.json(result.get("full_state", {}))
