"""Streamlit recruiter UI for RecruitSense.

Two tabs:
- **Single Screen** — score one resume against a JD, with a Plotly gauge,
  dimension bar chart, strengths/gaps, bias warnings, and the RAG context
  the scorer actually used.
- **Batch Screening** — upload up to 20 resumes, get a sortable leaderboard,
  a tier-distribution pie, and a CSV export.

The UI is purely a thin client over the FastAPI backend. Configure the API
base URL in the sidebar; defaults to ``http://localhost:8000``.
"""

from __future__ import annotations

import io
import json
import os
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Constants ----------------------------------------------------------------

DEFAULT_API_URL = os.environ.get("RECRUITSENSE_API_URL", "http://localhost:8000")
# Free-tier OpenRouter models only — no account credits required.
# Ordered fastest/most-reliable first. meta-llama-3.3-70b:free is intentionally
# omitted: it is frequently rate-limited (429) and produces zero fallback scores.
DEFAULT_MODEL = "openai/gpt-oss-120b:free"
MODEL_CHOICES = [
    "openai/gpt-oss-120b:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "z-ai/glm-4.5-air:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
]
MAX_BATCH_UI = 20  # UI limit; backend allows up to 50.
REQUEST_TIMEOUT_SECONDS = 600.0  # batch of 20 with cold cache can be slow.

TIER_COLORS = {
    "A": "#16a34a",  # green
    "B": "#0ea5e9",  # blue
    "C": "#f59e0b",  # amber
    "D": "#dc2626",  # red
}
DIMENSION_LABELS = {
    "skills_match": "Skills Match (35%)",
    "experience_relevance": "Experience Relevance (30%)",
    "education_and_certs": "Education & Certs (15%)",
    "project_impact": "Project Impact (10%)",
    "communication_and_polish": "Communication (10%)",
}

# --- Page config --------------------------------------------------------------

st.set_page_config(
    page_title="RecruitSense — LLM Resume Screener",
    page_icon=":briefcase:",
    layout="wide",
)


# --- Sidebar ------------------------------------------------------------------


def render_sidebar() -> tuple[str, str]:
    """Render the sidebar and return ``(api_url, model)``."""
    st.sidebar.title("RecruitSense")
    st.sidebar.caption("LLM-powered resume screening with RAG + bias detection.")

    api_url = st.sidebar.text_input(
        "Backend API URL",
        value=st.session_state.get("api_url", DEFAULT_API_URL),
        help="FastAPI backend base URL (no trailing slash).",
    ).rstrip("/")
    st.session_state["api_url"] = api_url

    model = st.sidebar.selectbox(
        "Scoring model",
        options=MODEL_CHOICES,
        index=MODEL_CHOICES.index(
            st.session_state.get("model", DEFAULT_MODEL)
            if st.session_state.get("model", DEFAULT_MODEL) in MODEL_CHOICES
            else DEFAULT_MODEL
        ),
        help="Free-tier OpenRouter model used for scoring. Sent to the backend with each request.",
    )
    st.session_state["model"] = model

    st.sidebar.divider()
    if st.sidebar.button("Check backend health"):
        _show_health(api_url)
    st.sidebar.caption("Backend: `/health` — Qdrant + Redis liveness")

    return api_url, model


def _show_health(api_url: str) -> None:
    """Poll ``/health`` and surface the result in the sidebar."""
    try:
        resp = httpx.get(f"{api_url}/health", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "unknown")
        if status == "ok":
            st.sidebar.success(f"Backend OK — Qdrant: {data['qdrant']}, Redis: {data['redis']}")
        else:
            st.sidebar.warning(
                f"Degraded — Qdrant: {data.get('qdrant')}, Redis: {data.get('redis')}"
            )
    except Exception as exc:
        st.sidebar.error(f"Backend unreachable: {exc}")


# --- JD form (shared between tabs) -------------------------------------------


def render_jd_form(key_prefix: str) -> dict[str, Any] | None:
    """Render the JD input form. Returns the JD dict on submit, else ``None``."""
    with st.form(key=f"{key_prefix}_jd_form"):
        col1, col2 = st.columns([2, 1])
        with col1:
            job_title = st.text_input("Job title *", key=f"{key_prefix}_title")
        with col2:
            company = st.text_input("Company", key=f"{key_prefix}_company")

        description = st.text_area(
            "Job description *",
            height=180,
            key=f"{key_prefix}_desc",
            help="Paste the full JD. Used by the LLM and the RAG sub-query generator.",
        )

        col3, col4 = st.columns(2)
        with col3:
            required_skills = st.text_input(
                "Required skills (comma-separated)",
                key=f"{key_prefix}_req_skills",
                placeholder="python, fastapi, postgres",
            )
        with col4:
            preferred_skills = st.text_input(
                "Preferred skills (comma-separated)",
                key=f"{key_prefix}_pref_skills",
                placeholder="kubernetes, terraform",
            )

        col5, col6 = st.columns(2)
        with col5:
            min_experience = st.number_input(
                "Minimum years of experience",
                min_value=0.0,
                max_value=50.0,
                value=0.0,
                step=0.5,
                key=f"{key_prefix}_min_exp",
            )
        with col6:
            education_level = st.selectbox(
                "Education level",
                options=["none", "high_school", "associate", "bachelor", "master", "phd"],
                index=0,
                key=f"{key_prefix}_edu",
            )

        submitted = st.form_submit_button("Screen", type="primary", use_container_width=True)

    if not submitted:
        return None

    if not job_title.strip() or len(description.strip()) < 10:
        st.error("Job title and a description (≥10 chars) are required.")
        return None

    return {
        "job_title": job_title.strip(),
        "company": company.strip(),
        "description": description.strip(),
        "required_skills": _split_csv(required_skills),
        "preferred_skills": _split_csv(preferred_skills),
        "min_experience_years": float(min_experience),
        "education_level": education_level,
    }


def _split_csv(text: str) -> list[str]:
    """Split a comma-separated input into a clean list, dropping blanks."""
    return [s.strip() for s in text.split(",") if s.strip()]


# --- Result rendering --------------------------------------------------------


def render_score(result: dict[str, Any]) -> None:
    """Render a single ``ScoreOutput`` payload."""
    tier = result.get("tier", "D")
    composite = float(result.get("composite_score", 0.0))
    confidence = float(result.get("confidence", 0.0))
    candidate = result.get("candidate_name", "Unknown")

    # Header row: candidate name + tier badge + recommended action
    st.subheader(f":bust_in_silhouette: {candidate}")
    badge_html = (
        f"<div style='display:inline-block;padding:6px 14px;border-radius:6px;"
        f"background:{TIER_COLORS.get(tier, '#888')};color:white;font-weight:700;"
        f"font-size:18px;margin-right:12px;'>Tier {tier}</div>"
        f"<span style='font-size:16px;'>Recommended action: "
        f"<b>{result.get('recommended_action', 'reject').replace('_', ' ').title()}</b></span>"
    )
    st.markdown(badge_html, unsafe_allow_html=True)

    # Gauge + confidence
    gauge_col, confidence_col = st.columns([2, 1])
    with gauge_col:
        st.plotly_chart(_composite_gauge(composite, tier), use_container_width=True)
    with confidence_col:
        st.metric("Composite score", f"{composite:.1f} / 100")
        st.metric("Model confidence", f"{confidence:.0%}")
        st.caption(f"Generated at {result.get('generated_at', '—')}")
        if result.get("model_used"):
            st.caption(f"Model: `{result['model_used']}`")

    # Dimension bars
    st.markdown("##### Per-dimension scores")
    st.plotly_chart(_dimension_bars(result.get("dimension_scores", {})), use_container_width=True)

    # Rationales (collapsible per dimension)
    with st.expander("Per-dimension rationales", expanded=False):
        for dim, label in DIMENSION_LABELS.items():
            ds = result.get("dimension_scores", {}).get(dim)
            if not ds:
                continue
            st.markdown(f"**{label} — {ds.get('score', 0):.1f}/10**")
            st.write(ds.get("rationale", ""))
            st.divider()

    # Strengths / Gaps side-by-side
    s_col, g_col = st.columns(2)
    with s_col:
        st.markdown("##### :white_check_mark: Top strengths")
        strengths = result.get("top_strengths", []) or ["_No standout strengths identified._"]
        for s in strengths:
            st.markdown(f"- {s}")
    with g_col:
        st.markdown("##### :warning: Key gaps")
        gaps = result.get("key_gaps", []) or ["_No critical gaps identified._"]
        for g in gaps:
            st.markdown(f"- {g}")

    # Bias warnings
    bias_flags = result.get("bias_flags", []) or []
    if bias_flags:
        st.markdown("##### :rotating_light: Potential bias signals")
        st.warning(
            "The following signals were detected on the resume **and were excluded "
            "from the LLM's scoring prompt**. Review them only to ensure fair "
            "downstream handling — do not use them to make a hiring decision."
        )
        for flag in bias_flags:
            st.markdown(
                f"<div style='background:#fff7ed;border-left:4px solid #f59e0b;"
                f"padding:8px 12px;margin:4px 0;border-radius:4px;'>{flag}</div>",
                unsafe_allow_html=True,
            )

    # RAG context (collapsed)
    rag_ctx = result.get("rag_context_used", "") or ""
    with st.expander(f":books: RAG context used ({len(rag_ctx):,} chars)", expanded=False):
        if rag_ctx:
            st.code(rag_ctx, language="markdown")
        else:
            st.caption("_No RAG context retrieved._")

    # Raw JSON download
    st.download_button(
        label=":inbox_tray: Download full JSON",
        data=json.dumps(result, indent=2, default=str),
        file_name=f"{candidate.replace(' ', '_').lower()}_score.json",
        mime="application/json",
    )


def _composite_gauge(score: float, tier: str) -> go.Figure:
    """Plotly gauge for the 0–100 composite score, colored by tier."""
    color = TIER_COLORS.get(tier, "#888")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Composite Score"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 55], "color": "#fee2e2"},
                    {"range": [55, 70], "color": "#fef3c7"},
                    {"range": [70, 85], "color": "#dbeafe"},
                    {"range": [85, 100], "color": "#dcfce7"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(height=280, margin={"l": 20, "r": 20, "t": 50, "b": 20})
    return fig


def _dimension_bars(dim_scores: dict[str, dict[str, Any]]) -> go.Figure:
    """Horizontal bar chart of per-dimension 0–10 scores."""
    rows = []
    for dim, label in DIMENSION_LABELS.items():
        ds = dim_scores.get(dim, {})
        rows.append({"Dimension": label, "Score": float(ds.get("score", 0.0))})
    df = pd.DataFrame(rows).iloc[::-1]  # reverse so highest-weighted dim is on top
    fig = px.bar(
        df,
        x="Score",
        y="Dimension",
        orientation="h",
        range_x=[0, 10],
        text="Score",
        color="Score",
        color_continuous_scale=["#dc2626", "#f59e0b", "#16a34a"],
        range_color=[0, 10],
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(
        height=280,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        coloraxis_showscale=False,
        yaxis_title=None,
        xaxis_title="Score (0–10)",
    )
    return fig


# --- API client --------------------------------------------------------------


def call_screen(
    api_url: str, jd: dict[str, Any], pdf_bytes: bytes, filename: str, model: str = ""
) -> dict[str, Any]:
    """POST one resume to the backend ``/screen`` endpoint."""
    files = {"resume": (filename, pdf_bytes, "application/pdf")}
    data = {"jd_json": json.dumps(jd), "model": model}
    resp = httpx.post(f"{api_url}/screen", data=data, files=files, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()


def call_batch(
    api_url: str, jd: dict[str, Any], pdfs: list[tuple[str, bytes]], model: str = ""
) -> dict[str, Any]:
    """POST many resumes to the backend ``/batch`` endpoint."""
    files = [("resumes", (name, content, "application/pdf")) for name, content in pdfs]
    data = {"jd_json": json.dumps(jd), "model": model}
    resp = httpx.post(f"{api_url}/batch", data=data, files=files, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()


def call_session_screen(
    api_url: str, jd: dict[str, Any], pdf_bytes: bytes, filename: str, model: str = ""
) -> dict[str, Any]:
    """Turn 1 — POST to ``/session/screen``; returns ``{session_id, score}``."""
    files = {"resume": (filename, pdf_bytes, "application/pdf")}
    data = {"jd_json": json.dumps(jd), "model": model}
    resp = httpx.post(
        f"{api_url}/session/screen", data=data, files=files, timeout=REQUEST_TIMEOUT_SECONDS
    )
    resp.raise_for_status()
    return resp.json()


def call_session_reweight(
    api_url: str, session_id: str, weight_overrides: dict[str, float]
) -> dict[str, Any]:
    """Turn 2 — POST to ``/session/{id}/reweight``; returns an updated ``ScoreOutput``."""
    resp = httpx.post(
        f"{api_url}/session/{session_id}/reweight",
        json={"weight_overrides": weight_overrides},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def call_session_compare(api_url: str, session_id_a: str, session_id_b: str) -> dict[str, Any]:
    """Turn 3 — POST to ``/session/compare``; returns a ``CompareResponse``."""
    resp = httpx.post(
        f"{api_url}/session/compare",
        json={"session_id_a": session_id_a, "session_id_b": session_id_b},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


# --- Tab 1: Single screen ----------------------------------------------------


def tab_single(api_url: str, model: str = "") -> None:
    """Render the single-resume screening tab."""
    st.markdown("### Screen one resume against a JD")
    jd = render_jd_form("single")

    resume_file = st.file_uploader(
        "Resume PDF *",
        type=["pdf"],
        accept_multiple_files=False,
        key="single_resume",
    )

    if jd is None:
        return  # form not yet submitted

    if resume_file is None:
        st.error("Please upload a resume PDF before clicking Screen.")
        return

    pdf_bytes = resume_file.getvalue()
    with st.spinner(f"Screening {resume_file.name}…"):
        try:
            result = call_screen(api_url, jd, pdf_bytes, resume_file.name, model=model)
        except httpx.HTTPStatusError as exc:
            st.error(f"Backend error {exc.response.status_code}: {exc.response.text}")
            return
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            return

    st.success("Done.")
    render_score(result)


# --- Tab 2: Batch screen -----------------------------------------------------


def tab_batch(api_url: str, model: str = "") -> None:
    """Render the batch screening tab."""
    st.markdown(f"### Batch screen up to {MAX_BATCH_UI} resumes")
    jd = render_jd_form("batch")

    resume_files = st.file_uploader(
        f"Resume PDFs * (max {MAX_BATCH_UI})",
        type=["pdf"],
        accept_multiple_files=True,
        key="batch_resumes",
    )

    if jd is None:
        return

    if not resume_files:
        st.error("Please upload at least one resume PDF.")
        return
    if len(resume_files) > MAX_BATCH_UI:
        st.error(f"Maximum {MAX_BATCH_UI} resumes per batch in the UI.")
        return

    pdfs = [(f.name, f.getvalue()) for f in resume_files]
    with st.spinner(f"Screening {len(pdfs)} resumes (this may take a minute)…"):
        try:
            result = call_batch(api_url, jd, pdfs, model=model)
        except httpx.HTTPStatusError as exc:
            st.error(f"Backend error {exc.response.status_code}: {exc.response.text}")
            return
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            return

    render_batch(result)


def render_batch(result: dict[str, Any]) -> None:
    """Render a ``BatchResult`` payload: KPIs, pie, leaderboard, CSV export."""
    total = int(result.get("total_resumes", 0))
    shortlisted = int(result.get("shortlisted_count", 0))
    duration = float(result.get("processing_time_seconds", 0.0))
    tier_dist: dict[str, int] = result.get("tier_distribution", {})
    candidates: list[dict[str, Any]] = result.get("ranked_candidates", [])

    st.success(
        f"Screened {total} resumes in {duration:.1f}s — {shortlisted} shortlisted (Tier A or B)."
    )

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total resumes", total)
    k2.metric("Shortlisted (A+B)", shortlisted)
    k3.metric("Shortlist rate", f"{(shortlisted / total * 100) if total else 0:.0f}%")
    k4.metric("Throughput", f"{(total / duration) if duration else 0:.1f} CV/s")

    # Pie chart
    pie_col, table_col = st.columns([1, 2])
    with pie_col:
        st.markdown("##### Tier distribution")
        st.plotly_chart(_tier_pie(tier_dist), use_container_width=True)
    with table_col:
        st.markdown("##### Leaderboard")
        df = _leaderboard_df(candidates)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Composite": st.column_config.ProgressColumn(
                    "Composite", min_value=0, max_value=100, format="%.1f"
                ),
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=1, format="%.0f%%"
                ),
            },
        )

    # CSV export
    csv_buf = io.StringIO()
    _full_csv(candidates).to_csv(csv_buf, index=False)
    st.download_button(
        label=":inbox_tray: Download full leaderboard CSV",
        data=csv_buf.getvalue(),
        file_name="recruitsense_leaderboard.csv",
        mime="text/csv",
    )

    # Per-candidate drilldown
    st.markdown("---")
    st.markdown("### Per-candidate detail")
    if not candidates:
        st.info("No candidates to display.")
        return
    names = [c.get("candidate_name", f"Candidate {i + 1}") for i, c in enumerate(candidates)]
    selected = st.selectbox("Select a candidate to inspect", options=names, index=0)
    chosen = next((c for c in candidates if c.get("candidate_name") == selected), candidates[0])
    render_score(chosen)


def _tier_pie(tier_dist: dict[str, int]) -> go.Figure:
    """Plotly pie chart of tier distribution, with brand colors per tier."""
    labels = ["A", "B", "C", "D"]
    values = [int(tier_dist.get(t, 0)) for t in labels]
    fig = px.pie(
        names=[f"Tier {t}" for t in labels],
        values=values,
        color=[f"Tier {t}" for t in labels],
        color_discrete_map={f"Tier {t}": TIER_COLORS[t] for t in labels},
        hole=0.45,
    )
    fig.update_traces(textinfo="value+label")
    fig.update_layout(height=320, margin={"l": 10, "r": 10, "t": 20, "b": 10})
    return fig


def _leaderboard_df(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the compact leaderboard dataframe shown in the Batch tab."""
    rows = []
    for i, c in enumerate(candidates, start=1):
        rows.append(
            {
                "Rank": i,
                "Candidate": c.get("candidate_name", "Unknown"),
                "Tier": c.get("tier", "D"),
                "Composite": float(c.get("composite_score", 0.0)),
                "Action": c.get("recommended_action", "reject").replace("_", " ").title(),
                "Confidence": float(c.get("confidence", 0.0)),
                "Bias flags": len(c.get("bias_flags", []) or []),
            }
        )
    return pd.DataFrame(rows)


def _full_csv(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    """Wider dataframe used for CSV export (one row per candidate, all dimensions)."""
    rows = []
    for i, c in enumerate(candidates, start=1):
        dims = c.get("dimension_scores", {}) or {}
        row = {
            "rank": i,
            "candidate": c.get("candidate_name", "Unknown"),
            "tier": c.get("tier", "D"),
            "composite_score": float(c.get("composite_score", 0.0)),
            "recommended_action": c.get("recommended_action", "reject"),
            "confidence": float(c.get("confidence", 0.0)),
            "bias_flag_count": len(c.get("bias_flags", []) or []),
            "bias_flags": "; ".join(c.get("bias_flags", []) or []),
            "top_strengths": " | ".join(c.get("top_strengths", []) or []),
            "key_gaps": " | ".join(c.get("key_gaps", []) or []),
        }
        for dim in DIMENSION_LABELS:
            row[f"{dim}_score"] = float(dims.get(dim, {}).get("score", 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


# --- Tab 3: Sessions (stateful LangGraph) ------------------------------------


def tab_session(api_url: str, model: str = "") -> None:
    """Render the stateful session tab — Turn 1 screen, Turn 2 reweight, Turn 3 compare.

    Screened candidates are kept in ``st.session_state['sessions']`` so their
    backend ``session_id`` survives Streamlit reruns and can be reused for
    reweighting and comparison without re-uploading any resume.
    """
    st.markdown("### Stateful screening sessions")
    st.caption(
        "Demonstrates LangGraph's checkpointed state. Screen a candidate once (Turn 1), "
        "then reweight or compare with **no further LLM calls** — the dimension scores "
        "are read back from the saved graph state."
    )

    if "sessions" not in st.session_state:
        st.session_state["sessions"] = []  # list of {session_id, candidate_name, score}

    turn1, turn2, turn3 = st.tabs(["1️⃣ Screen (saves state)", "2️⃣ Reweight", "3️⃣ Compare"])

    with turn1:
        _session_turn1(api_url, model)
    with turn2:
        _session_turn2(api_url)
    with turn3:
        _session_turn3(api_url)


def _session_label(s: dict[str, Any]) -> str:
    """Human-readable dropdown label for a saved session."""
    score = s["score"]
    short_id = s["session_id"][:8]
    return f"{s['candidate_name']} — Tier {score.get('tier', '?')} ({short_id})"


def _session_turn1(api_url: str, model: str) -> None:
    """Turn 1 — run the full pipeline and persist state under a session_id."""
    jd = render_jd_form("session")
    resume_file = st.file_uploader(
        "Resume PDF *",
        type=["pdf"],
        accept_multiple_files=False,
        key="session_resume",
    )

    if jd is None:
        return
    if resume_file is None:
        st.error("Please upload a resume PDF before clicking Screen.")
        return

    pdf_bytes = resume_file.getvalue()
    with st.spinner(f"Screening {resume_file.name} and saving session…"):
        try:
            payload = call_session_screen(api_url, jd, pdf_bytes, resume_file.name, model=model)
        except httpx.HTTPStatusError as exc:
            st.error(f"Backend error {exc.response.status_code}: {exc.response.text}")
            return
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            return

    session_id = payload["session_id"]
    score = payload["score"]
    st.session_state["sessions"].append(
        {
            "session_id": session_id,
            "candidate_name": score.get("candidate_name", "Unknown"),
            "score": score,
        }
    )
    st.success(f"Session saved — `{session_id}`. Now available in Reweight and Compare tabs.")
    render_score(score)


def _session_turn2(api_url: str) -> None:
    """Turn 2 — recompute the composite with custom weights (no LLM call)."""
    sessions = st.session_state["sessions"]
    if not sessions:
        st.info("No saved sessions yet. Screen a candidate in Turn 1 first.")
        return

    idx = st.selectbox(
        "Select a saved session to reweight",
        options=list(range(len(sessions))),
        format_func=lambda i: _session_label(sessions[i]),
        key="reweight_pick",
    )
    chosen = sessions[idx]

    st.markdown("##### Adjust dimension weights")
    st.caption("Weights are normalized to sum to 1.0 before being sent to the backend.")

    defaults = {
        "skills_match": 35,
        "experience_relevance": 30,
        "education_and_certs": 15,
        "project_impact": 10,
        "communication_and_polish": 10,
    }
    raw_weights: dict[str, int] = {}
    cols = st.columns(len(defaults))
    for col, (dim, default) in zip(cols, defaults.items(), strict=True):
        with col:
            raw_weights[dim] = st.slider(
                DIMENSION_LABELS[dim].split(" (")[0],
                min_value=0,
                max_value=100,
                value=default,
                step=5,
                key=f"rw_{dim}",
            )

    total = sum(raw_weights.values())
    if total == 0:
        st.error("At least one weight must be greater than zero.")
        return

    normalized = {dim: round(v / total, 4) for dim, v in raw_weights.items()}
    # Correct any rounding drift so the sum is exactly 1.0.
    drift = round(1.0 - sum(normalized.values()), 4)
    first_dim = next(iter(normalized))
    normalized[first_dim] = round(normalized[first_dim] + drift, 4)

    st.caption(
        "Normalized weights sent to backend: "
        + ", ".join(f"{dim}={w:.2f}" for dim, w in normalized.items())
    )

    if st.button("Recompute score", type="primary", key="reweight_btn"):
        with st.spinner("Recomputing from saved state (no LLM call)…"):
            try:
                updated = call_session_reweight(api_url, chosen["session_id"], normalized)
            except httpx.HTTPStatusError as exc:
                st.error(f"Backend error {exc.response.status_code}: {exc.response.text}")
                return
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                return

        old = chosen["score"]
        c1, c2 = st.columns(2)
        c1.metric(
            "Original composite",
            f"{old.get('composite_score', 0):.1f}",
            help=f"Tier {old.get('tier')}",
        )
        c2.metric(
            "Reweighted composite",
            f"{updated.get('composite_score', 0):.1f}",
            delta=round(updated.get("composite_score", 0) - old.get("composite_score", 0), 1),
            help=f"Tier {updated.get('tier')}",
        )
        st.divider()
        render_score(updated)


def _session_turn3(api_url: str) -> None:
    """Turn 3 — compare two saved sessions side by side (no LLM call)."""
    sessions = st.session_state["sessions"]
    if len(sessions) < 2:
        st.info("Need at least two saved sessions to compare. Screen another candidate in Turn 1.")
        return

    c1, c2 = st.columns(2)
    with c1:
        idx_a = st.selectbox(
            "Candidate A",
            options=list(range(len(sessions))),
            format_func=lambda i: _session_label(sessions[i]),
            key="cmp_a",
        )
    with c2:
        idx_b = st.selectbox(
            "Candidate B",
            options=list(range(len(sessions))),
            format_func=lambda i: _session_label(sessions[i]),
            index=1,
            key="cmp_b",
        )

    if idx_a == idx_b:
        st.warning("Select two different candidates.")
        return

    if st.button("Compare", type="primary", key="compare_btn"):
        with st.spinner("Comparing saved states (no LLM call)…"):
            try:
                cmp = call_session_compare(
                    api_url, sessions[idx_a]["session_id"], sessions[idx_b]["session_id"]
                )
            except httpx.HTTPStatusError as exc:
                st.error(f"Backend error {exc.response.status_code}: {exc.response.text}")
                return
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                return

        _render_comparison(cmp)


def _render_comparison(cmp: dict[str, Any]) -> None:
    """Render a ``CompareResponse`` — winner banner, score delta, per-dimension table."""
    a = cmp["candidate_a"]
    b = cmp["candidate_b"]
    winner = cmp.get("overall_winner", "tie")
    delta = cmp.get("score_delta", 0.0)

    if winner == "tie":
        st.info(f"**Tie** — both candidates scored {a.get('composite_score', 0):.1f}.")
    else:
        st.success(f"**Overall winner: {winner}** (composite delta {abs(delta):.1f} points)")

    c1, c2 = st.columns(2)
    c1.metric(
        a.get("candidate_name", "A"),
        f"{a.get('composite_score', 0):.1f}",
        help=f"Tier {a.get('tier')}",
    )
    c2.metric(
        b.get("candidate_name", "B"),
        f"{b.get('composite_score', 0):.1f}",
        help=f"Tier {b.get('tier')}",
    )

    st.markdown("##### Per-dimension comparison")
    rows = []
    for dim, comp in cmp.get("dimension_comparison", {}).items():
        win = comp.get("winner", "tie")
        win_name = (
            a.get("candidate_name", "A")
            if win == "A"
            else (b.get("candidate_name", "B") if win == "B" else "tie")
        )
        rows.append(
            {
                "Dimension": DIMENSION_LABELS.get(dim, dim).split(" (")[0],
                a.get("candidate_name", "A"): comp.get("candidate_a_score", 0.0),
                b.get("candidate_name", "B"): comp.get("candidate_b_score", 0.0),
                "Winner": win_name,
                "Δ": comp.get("delta", 0.0),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Full detail — Candidate A", expanded=False):
        render_score(a)
    with st.expander("Full detail — Candidate B", expanded=False):
        render_score(b)


# --- Main --------------------------------------------------------------------


def main() -> None:
    """Streamlit entrypoint — mounts the sidebar and the three tabs."""
    api_url, model = render_sidebar()

    st.title(":briefcase: RecruitSense")
    st.caption(
        "LLM-powered resume screening — RAG (Qdrant + BGE-large), "
        "LangGraph multi-agent orchestration, and bias detection."
    )

    tab1, tab2, tab3 = st.tabs(
        [":mag: Single Screen", ":bar_chart: Batch Screening", ":repeat: Sessions"]
    )
    with tab1:
        tab_single(api_url, model=model)
    with tab2:
        tab_batch(api_url, model=model)
    with tab3:
        tab_session(api_url, model=model)


if __name__ == "__main__":
    main()
