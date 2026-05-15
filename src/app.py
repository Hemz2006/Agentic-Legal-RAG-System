import html

import streamlit as st

from config import EMBEDDING_MODEL, LEGAL_DATASET_NAME, RETRIEVER_BACKEND
from rag_pipeline import run_agentic_pipeline
from retriever import build_retriever


APP_TITLE = "Agentic Legal RAG"
TOP_CASES = 3


st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #102033;
        --ink: #17202a;
        --muted: #5f6b7a;
        --line: #d8dee8;
        --panel: #ffffff;
        --soft: #f6f8fb;
        --accent: #2f6bff;
        --success: #0f766e;
    }

    .stApp {
        background: var(--soft);
        color: var(--ink);
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.4rem;
        max-width: 1280px;
    }

    h1, h2, h3 {
        color: var(--navy);
        letter-spacing: 0;
    }

    .app-header {
        border-bottom: 1px solid var(--line);
        padding-bottom: 1.1rem;
        margin-bottom: 1.2rem;
    }

    .eyebrow {
        color: var(--accent);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08rem;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .subtitle {
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.55;
        max-width: 860px;
        margin-top: 0.3rem;
    }

    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.95rem 1rem;
    }

    div[data-testid="stTextArea"] textarea {
        border-radius: 8px;
        border-color: var(--line);
        min-height: 130px;
    }

    .stButton > button {
        background: var(--navy);
        color: white;
        border: 1px solid var(--navy);
        border-radius: 8px;
        font-weight: 700;
        height: 3rem;
        width: 100%;
    }

    .stButton > button:hover {
        background: #1a3658;
        border-color: #1a3658;
        color: white;
    }

    .case-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.9rem;
    }

    .case-title {
        color: var(--navy);
        font-weight: 800;
        font-size: 1rem;
        margin-bottom: 0.2rem;
    }

    .case-meta {
        color: var(--muted);
        font-size: 0.86rem;
        margin-bottom: 0.7rem;
    }

    .case-text {
        color: var(--ink);
        font-size: 0.94rem;
        line-height: 1.5;
    }

    .notice {
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.45;
        border-top: 1px solid var(--line);
        margin-top: 1rem;
        padding-top: 0.8rem;
    }

    .agent-step {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.85rem 0.95rem;
        min-height: 92px;
    }

    .agent-step-title {
        color: var(--navy);
        font-size: 0.9rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .agent-step-detail {
        color: var(--muted);
        font-size: 0.85rem;
        line-height: 1.4;
    }

    .status-pill {
        display: inline-block;
        border-radius: 999px;
        background: #eaf1ff;
        color: #1d4ed8;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 0.22rem 0.55rem;
        margin-top: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_system():
    return build_retriever()


def preview_text(text: str, limit: int = 900) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return html.escape(cleaned)
    return html.escape(f"{cleaned[:limit].rstrip()}...")


def score_label(score: float) -> str:
    if score >= 0.55:
        return "High relevance"
    if score >= 0.35:
        return "Moderate relevance"
    return "Low relevance"


with st.sidebar:
    st.markdown("### Workspace")
    st.caption("Indian judiciary research assistant for top-judgment retrieval and FIRAC analysis.")
    st.divider()
    st.markdown("### Retrieval")
    st.write(f"Cases returned for analysis: {TOP_CASES}")
    st.write(f"Backend: {RETRIEVER_BACKEND}")
    st.write(f"Embedding model: {EMBEDDING_MODEL if RETRIEVER_BACKEND == 'dense' else 'TF-IDF local fallback'}")
    st.write(f"Dataset: {LEGAL_DATASET_NAME}")
    st.write("Search method: FAISS semantic similarity")
    st.write("Agents: query, retrieval, decision, analyzer")
    st.write("Answer mode: source-grounded FIRAC analysis")
    st.divider()
    st.caption("This tool provides research assistance only. It is not a substitute for qualified legal advice.")


st.markdown(
    """
    <div class="app-header">
        <div class="eyebrow">Enterprise Legal Search</div>
        <h1>Agentic Legal RAG System</h1>
        <div class="subtitle">
            Search Indian court judgments and Supreme Court-style case law with keywords or natural language,
            review the top three most relevant judgments, and receive a FIRAC analysis grounded in retrieved sources.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    index, texts = load_system()
except Exception as exc:
    st.error(f"Unable to load the legal search index: {exc}")
    st.stop()

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Indexed documents", f"{len(texts):,}")
metric_col2.metric("Similar cases shown", TOP_CASES)
metric_col3.metric("Retriever backend", RETRIEVER_BACKEND)

st.markdown("### Agentic Workflow")
workflow_cols = st.columns(4)
workflow_items = [
    ("Query Agent", "Expands the user query into related legal search intents."),
    ("Retrieval Agent", "Searches the legal corpus with the configured retrieval backend."),
    ("Decision Agent", "Checks whether retrieved evidence is strong enough, then refines if needed."),
    ("Analyzer Agent", "Selects the top judgments and structures evidence for FIRAC reasoning."),
]
for col, (title, detail) in zip(workflow_cols, workflow_items):
    with col:
        st.markdown(
            f"""
            <div class="agent-step">
                <div class="agent-step-title">{title}</div>
                <div class="agent-step-detail">{detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("### Case Search")
query = st.text_area(
    "Enter legal keywords, facts, provisions, or a question",
    placeholder=(
        "Example: anticipatory bail Supreme Court guidelines, IPC 302 murder conviction, "
        "dowry harassment cruelty, property fraud cheating"
    ),
    label_visibility="visible",
)

action_col, hint_col = st.columns([1, 3])
with action_col:
    run = st.button("Analyze Cases", type="primary")
with hint_col:
    st.caption("Use a full sentence for best results. Include facts, legal issue, jurisdiction, or remedy if known.")

if run:
    if not query.strip():
        st.warning("Enter a legal query before running the analysis.")
    else:
        with st.spinner("Searching the case database and preparing legal analysis..."):
            result = run_agentic_pipeline(query, index, texts, top_k=TOP_CASES)
            retrieved_docs = result["analysis"]["top_cases"]
            answer = result["answer"]
            decision = result["decision"]

        if not retrieved_docs:
            st.error("No relevant cases were found for this query.")
        else:
            st.markdown("### Analysis Run")
            run_cols = st.columns(4)
            run_cols[0].metric("Expanded queries", len(result["expanded_queries"]))
            run_cols[1].metric("Retrieved candidates", len(result["retrieved"]))
            run_cols[2].metric("Score range", result["analysis"]["score_range"])
            run_cols[3].metric("Decision", "Sufficient" if decision["sufficient"] else "Limited")

            st.markdown(
                f"""
                <div class="notice">
                    <strong>Decision agent:</strong> {html.escape(decision["reason"])}
                    <span class="status-pill">{"Refined search used" if result["refined"] else "Initial search sufficient"}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Agent trace", expanded=True):
                st.write("**Expanded queries:**")
                for expanded_query in result["expanded_queries"]:
                    st.write(f"- {expanded_query}")
                st.write(
                    f"**Decision:** {decision['reason']} "
                    f"(best score: {decision['best_score']:.3f})"
                )
                st.write(f"**Refinement triggered:** {result['refined']}")
                st.write(f"**Analyzer selected:** {result['analysis']['case_count']} top cases")

            st.markdown("### Top 3 Relevant Judgments")
            for case_number, (doc, score) in enumerate(retrieved_docs, start=1):
                st.markdown(
                    f"""
                    <div class="case-card">
                        <div class="case-title">Judgment {case_number}</div>
                        <div class="case-meta">Similarity score: {score:.3f} | {score_label(score)}</div>
                        <div class="case-text">{preview_text(doc)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander(f"Open full Judgment {case_number} text"):
                    st.text(doc)

            with st.expander("All retrieved candidates with similarity scores"):
                for source_number, (doc, score) in enumerate(result["retrieved"], start=1):
                    st.markdown(f"**Judgment candidate {source_number} — similarity: {score:.3f}**")
                    st.text(doc)

            st.markdown("### FIRAC Analysis")
            st.markdown(answer)

            st.markdown(
                """
                <div class="notice">
                    The FIRAC analysis is generated from the retrieved judgment text only.
                    Review the original judgment materials before relying on the analysis.
                </div>
                """,
                unsafe_allow_html=True,
            )
