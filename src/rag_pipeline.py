"""Agentic RAG pipeline for legal case analysis."""
import logging
import re
from typing import List, Tuple

from openai import OpenAI, OpenAIError

from agents import AnalyzerAgent, DecisionAgent, QueryExpansionAgent, RetrievalAgent
from config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, TOP_K
from retriever import build_retriever, retrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to a .env file at the project root."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


def expand_query(query: str) -> List[str]:
    """Generate query variants to improve recall."""
    return QueryExpansionAgent().run(query)


def multi_query_retrieve(
    query: str,
    index,
    texts: List[str],
    top_k: int = TOP_K,
) -> List[Tuple[str, float]]:
    """Run retrieval over multiple query variants and merge by best score per doc."""
    retrieval_agent = RetrievalAgent(retrieve_fn=retrieve)
    return retrieval_agent.run(expand_query(query), index, texts, top_k=top_k)[:top_k]


def refinement_queries(query: str) -> List[str]:
    """Create fallback queries when first-pass retrieval looks weak."""
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    return [
        f"similar facts and legal issues for {cleaned_query}",
        f"court reasoning and legal principles in cases about {cleaned_query}",
        f"precedents involving {cleaned_query}",
    ]


def run_agentic_pipeline(query: str, index, texts: List[str], top_k: int = TOP_K) -> dict:
    """Run the explainable agentic retrieval and analysis pipeline."""
    query_agent = QueryExpansionAgent()
    retrieval_agent = RetrievalAgent(retrieve_fn=retrieve)
    decision_agent = DecisionAgent(min_cases=3)
    analyzer_agent = AnalyzerAgent(top_n=3)

    expanded_queries = query_agent.run(query)
    retrieved = retrieval_agent.run(expanded_queries, index, texts, top_k=top_k)
    decision = decision_agent.run(retrieved)
    refined = False

    if not decision["sufficient"]:
        refined = True
        refined_queries = expanded_queries + refinement_queries(query)
        retrieved = retrieval_agent.run(refined_queries, index, texts, top_k=top_k + 3)
        decision = decision_agent.run(retrieved)
        expanded_queries = refined_queries

    analysis = analyzer_agent.run(query, retrieved)
    answer = generate_answer(query, analysis["top_cases"])

    return {
        "query": query,
        "expanded_queries": expanded_queries,
        "retrieved": retrieved,
        "analysis": analysis,
        "decision": decision,
        "refined": refined,
        "answer": answer,
    }


SYSTEM_PROMPT = """You are a professional Indian legal research assistant analyzing retrieved Supreme Court and court judgment excerpts.

Your job:
1. Read the top retrieved case documents carefully.
2. Treat the retrieved sources as the only available authorities.
3. Select and compare the top three most relevant judgments.
4. Produce a FIRAC-form legal analysis grounded only in the retrieved sources.
5. Cite sources you use, e.g. [Judgment 1], [Judgment 2].

Format your response as:
**Top 3 Relevant Judgments:** A numbered list. For each judgment, state why it is relevant.
**F - Facts:** Briefly state only the most important facts found in the retrieved judgments.
**I - Issues:** Identify the top 3 legal issues. Each issue must start with "Whether".
**R - Rules:** List the provisions, Acts, sections, constitutional articles, and legal principles found in the retrieved judgments. If a source does not state a provision, say "Not expressly stated in retrieved text."
**A - Analysis:** Apply the rules to the facts and issues in detail, citing [Judgment 1], [Judgment 2], etc.
**C - Conclusion:** State the court's conclusion/final judgment as reflected in the retrieved text. If the final holding is not available in the excerpt, say so clearly.
**Research Note:** State limitations and remind the user this is research assistance, not legal advice.

Only reply "Not enough data available in the retrieved documents." if the documents are TRULY unrelated (e.g., a query about fraud and all sources are about cooking recipes). If there is ANY relevant information, use it — don't refuse.

Do not invent facts not in the sources. Do not add legal advice beyond what the sources state."""


def generate_answer(query: str, retrieved: List[Tuple[str, float]]) -> str:
    """Call the LLM with the retrieved context."""
    if not retrieved:
        return "Not enough data available in the retrieved documents."

    fallback_answer = generate_local_answer(query, retrieved)
    if not OPENAI_API_KEY:
        return fallback_answer

    top_retrieved = retrieved[:3]
    context_blocks = [
        f"[Judgment {i+1}] (similarity={score:.3f})\n{doc}"
        for i, (doc, score) in enumerate(top_retrieved)
    ]
    context = "\n\n".join(context_blocks)

    user_prompt = f"""Top Retrieved Cases:
{context}

User Keywords / Query:
{query}

Select the top three most relevant judgments and provide a FIRAC analysis using only the judgment text above."""

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=LLM_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
    except (OpenAIError, RuntimeError) as e:
        logger.error(f"LLM error: {e}")
        return fallback_answer


def generate_local_answer(query: str, retrieved: List[Tuple[str, float]]) -> str:
    """Create a readable FIRAC research brief without calling an external LLM."""
    top_retrieved = retrieved[:3]
    judgment_lines = []
    for index, (doc, score) in enumerate(top_retrieved, start=1):
        summary = _clean_snippet(_important_excerpt(doc, query, limit=520))
        judgment_lines.append(
            f"{index}. **[Judgment {index}]** similarity={score:.3f}. {summary}..."
        )

    best_score = top_retrieved[0][1] if top_retrieved else 0.0
    facts = _facts_from_judgment(top_retrieved[0][0], query) if top_retrieved else []
    rules = _rules_from_judgments([doc for doc, _ in top_retrieved])
    analysis = _analysis_from_judgments(query, top_retrieved)
    conclusion = _conclusion_from_judgment(top_retrieved[0][0]) if top_retrieved else "No conclusion available."

    return f"""**Top 3 Relevant Judgments:**
{chr(10).join(judgment_lines)}

**F - Facts:**
{_format_bullets(facts)}

**I - Issues:**
1. Whether the legal principles in the retrieved judgments address `{query}`.
2. Whether the material facts in the top-ranked judgment are comparable to the user's search issue.
3. Whether the provisions, sections, or judicial principles identified in the retrieved judgments support the final conclusion.

**R - Rules:**
{rules}

**A - Analysis:**
{analysis}

**C - Conclusion:**
{_clean_snippet(conclusion, limit=900)}

**Research Note:**
This local FIRAC brief extracts readable passages from the retrieved judgments. It does not infer missing facts or holdings. Add `OPENAI_API_KEY` to `.env` for a fully written LLM FIRAC. This tool provides legal research assistance only, not legal advice."""


def _clean_text(text: str) -> str:
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b([A-Za-z])\s+(?=[A-Za-z]\b)", r"\1", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text.strip()


def _clean_snippet(text: str, limit: int = 650) -> str:
    cleaned = _clean_text(text)
    if len(cleaned) <= limit:
        return cleaned
    cut = cleaned[:limit].rsplit(" ", 1)[0]
    return cut.strip()


def _sentences(text: str) -> List[str]:
    cleaned = _clean_text(text)
    protected = {
        "No.": "No<dot>",
        "Nos.": "Nos<dot>",
        "Mr.": "Mr<dot>",
        "Mrs.": "Mrs<dot>",
        "Dr.": "Dr<dot>",
        "J.": "J<dot>",
        "Ltd.": "Ltd<dot>",
        "vs.": "vs<dot>",
        "v.": "v<dot>",
    }
    for original, replacement in protected.items():
        cleaned = cleaned.replace(original, replacement)
    raw = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
    raw = [part.replace("<dot>", ".") for part in raw]
    return [
        sentence for sentence in raw
        if _is_complete_sentence(sentence)
    ]


def _is_complete_sentence(sentence: str) -> bool:
    if len(sentence) < 55 or len(sentence.split()) < 9:
        return False
    bad_endings = (" no.", " nos.", " v.", " vs.", " mr.", " dr.", "crl.", "appeal no.")
    if sentence.lower().endswith(bad_endings):
        return False
    return True


def _query_terms(query: str) -> set[str]:
    return {
        term.lower()
        for term in re.findall(r"[a-zA-Z0-9]+", query)
        if len(term) > 2
    }


def _important_excerpt(text: str, query: str, limit: int = 520) -> str:
    terms = _query_terms(query)
    ranked = sorted(_sentences(text)[:160], key=lambda sentence: _sentence_score(sentence, terms), reverse=True)
    selected = ranked[0] if ranked else " ".join(text.split())
    return _clean_snippet(selected, limit=limit)


def _sentence_score(sentence: str, terms: set[str]) -> int:
    lower = sentence.lower()
    score = sum(2 for term in terms if term in lower)
    legal_markers = (
        "section", "article", "ipc", "crpc", "act", "court", "held",
        "appeal", "conviction", "sentence", "petitioner", "respondent",
    )
    score += sum(1 for marker in legal_markers if marker in lower)
    return score


def _facts_from_judgment(text: str, query: str) -> str:
    terms = _query_terms(query)
    fact_markers = (
        "facts", "occurrence", "alleged", "accused", "appellant", "respondent",
        "prosecution", "trial", "convicted", "charged", "petition", "appeal",
    )
    candidates = [
        sentence for sentence in _sentences(text)[:80]
        if any(marker in sentence.lower() for marker in fact_markers)
    ]
    if not candidates:
        candidates = [_important_excerpt(text, query, limit=700)]
    candidates = sorted(
        candidates,
        key=lambda sentence: _sentence_score(sentence, terms),
        reverse=True,
    )
    return [_clean_snippet(sentence, limit=320) for sentence in candidates[:4]]


def _format_bullets(items: List[str]) -> str:
    if not items:
        return "- No clear facts were extracted from the retrieved judgment text."
    return "\n".join(f"- {item}" for item in items)


def _rules_from_judgments(judgments: List[str]) -> str:
    patterns = [
        r"(?:section|s\.|sec\.)\s+\d+[A-Za-z-]*(?:\(\d+\))*",
        r"article\s+\d+[A-Za-z-]*",
        r"Indian Penal Code|IPC|CrPC|Code of Criminal Procedure|Constitution of India",
        r"[A-Z][A-Za-z ]+ Act,?\s+\d{4}",
    ]
    found: list[str] = []
    for judgment in judgments:
        for pattern in patterns:
            found.extend(re.findall(pattern, judgment, flags=re.IGNORECASE))

    unique = []
    for item in found:
        normalized = " ".join(item.split())
        if normalized.lower() not in {existing.lower() for existing in unique}:
            unique.append(normalized)

    if not unique:
        return "Not expressly stated in retrieved text. Review the full judgments shown above for provisions and principles."
    return "\n".join(f"- {item}" for item in unique[:12])


def _analysis_from_judgments(query: str, retrieved: List[Tuple[str, float]]) -> str:
    lines = []
    for index, (doc, score) in enumerate(retrieved, start=1):
        excerpt = _clean_snippet(_important_excerpt(doc, query, limit=650))
        lines.append(
            f"- **[Judgment {index}]** matches the query with similarity {score:.3f}. "
            f"Relevant passage: \"{excerpt}...\""
        )
    return "\n".join(lines)


def _conclusion_from_judgment(text: str) -> str:
    sentences = _sentences(text)
    conclusion_markers = ("appeal is allowed", "appeal allowed", "appeal dismissed", "petition dismissed", "petition allowed", "conviction", "sentence", "order")
    for sentence in reversed(sentences[-80:]):
        if any(marker in sentence.lower() for marker in conclusion_markers):
            return sentence[:900]
    return "The final holding is not clearly available from the retrieved text excerpt. Review the full judgment displayed above."


def run_pipeline(query: str) -> dict:
    """End-to-end RAG run."""
    index, texts = build_retriever()
    return run_agentic_pipeline(query, index, texts)


if __name__ == "__main__":
    query = "Explain fraud-related legal cases based on available documents"
    result = run_pipeline(query)

    print("\n🔍 Retrieved Sources:\n" + "=" * 60)
    for i, (doc, score) in enumerate(result["retrieved"], 1):
        print(f"\n[Source {i}] score={score:.4f}\n{doc[:300]}...")

    print("\n\n🧠 Final AI Answer:\n" + "=" * 60)
    print(result["answer"])
