"""Legacy-compatible orchestration that now delegates to the unified pipeline.

Kept so `app.py` and `evaluation.py` imports keep working, but the heavy
generation/extractive logic has moved into `trace_law` (single source of truth).
"""
import logging
from typing import List, Tuple

from agents import AnalyzerAgent, DecisionAgent, QueryExpansionAgent, RetrievalAgent
from config import LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY, TOP_K
from retriever import build_retriever, retrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a professional Indian legal research assistant analyzing retrieved
Supreme Court and High Court judgment excerpts. Treat the retrieved sources as the only authorities.
Produce a FIRAC analysis (Facts, Issues, Rule, Application, Conclusion) grounded ONLY in the sources,
citing them as [Judgment 1], [Judgment 2]. Never invent facts or citations. This is research
assistance, not legal advice."""


def expand_query(query: str) -> List[str]:
    return QueryExpansionAgent().run(query)


def multi_query_retrieve(query: str, index, texts: List[str], top_k: int = TOP_K) -> List[Tuple[str, float]]:
    """Retrieve over query variants and fuse with RRF (text used as the doc id)."""
    agent = RetrievalAgent(retrieve_fn=retrieve)  # fusion='rrf' by default
    return agent.run(expand_query(query), index, texts, top_k=top_k)[:top_k]


def generate_answer(query: str, retrieved: List[Tuple[str, float]]) -> str:
    """Generate a FIRAC answer via the unified generator (OpenAI/local/extractive)."""
    from trace_law.generation import get_generator
    if OPENAI_API_KEY:
        gen = get_generator("openai", model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    else:
        gen = get_generator("extractive")
    return gen.generate(query, retrieved)


def run_agentic_pipeline(query: str, index, texts: List[str], top_k: int = TOP_K) -> dict:
    """Run the full TRACE-Law pipeline using the given engine, in an app-friendly shape."""
    import pipeline

    id_to_text = {t: t for t in texts}  # in the app, the text is its own id

    def dense(q, k):
        return [(t, s) for t, s in retrieve(q, index, texts, top_k=k)]

    res = pipeline.answer(query, [dense], id_to_text, top_k_retrieve=top_k)

    top_cases = list(res.evidence)
    analyzer = AnalyzerAgent(top_n=len(top_cases) or 3)
    analysis = analyzer.run(query, top_cases)
    decision = {
        "sufficient": not res.abstained,
        "best_score": res.reliability.get("components", {}).get("retrieval", 0.0),
        "reason": "; ".join(res.reliability.get("reasons", [])) or "Evidence sufficient.",
    }
    retrieved = list(res.reranked)[:top_k] or top_cases
    return {
        "query": query,
        "expanded_queries": res.expansions,
        "retrieved": retrieved,
        "analysis": analysis,
        "decision": decision,
        "refined": False,
        "temporal": res.temporal,
        "verification": res.verification,
        "reliability": res.reliability,
        "answer": res.answer,
    }


def run_pipeline(query: str) -> dict:
    index, texts = build_retriever()
    return run_agentic_pipeline(query, index, texts)


if __name__ == "__main__":
    result = run_pipeline("fraud related legal cases")
    print(result["answer"])
