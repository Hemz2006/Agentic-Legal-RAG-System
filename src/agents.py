"""Agent helpers for the LegalAssist RAG workflow."""
from dataclasses import dataclass
from typing import Callable, List, Tuple


RetrievedCase = Tuple[str, float]


@dataclass(frozen=True)
class QueryExpansionAgent:
    """Creates retrieval-focused variants of the user's legal query."""

    def run(self, query: str) -> List[str]:
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        return [
            cleaned_query,
            f"Supreme Court judgment on {cleaned_query}",
            f"Indian case law related to {cleaned_query}",
            f"judgment reasoning provisions sections for {cleaned_query}",
        ]


@dataclass(frozen=True)
class RetrievalAgent:
    """Runs semantic retrieval for each expanded query."""

    retrieve_fn: Callable

    def run(self, queries: List[str], index, texts: List[str], top_k: int) -> List[RetrievedCase]:
        best_matches: dict[str, float] = {}

        for query in queries:
            for document, score in self.retrieve_fn(query, index, texts, top_k=top_k):
                if document not in best_matches or score > best_matches[document]:
                    best_matches[document] = score

        return sorted(best_matches.items(), key=lambda item: item[1], reverse=True)


@dataclass(frozen=True)
class DecisionAgent:
    """Checks whether retrieved evidence is strong enough for legal analysis."""

    min_cases: int = 3
    min_best_score: float = 0.28

    def run(self, retrieved_cases: List[RetrievedCase]) -> dict:
        if not retrieved_cases:
            return {
                "sufficient": False,
                "reason": "No semantically similar cases were retrieved.",
                "best_score": 0.0,
            }

        best_score = retrieved_cases[0][1]
        sufficient = len(retrieved_cases) >= self.min_cases and best_score >= self.min_best_score
        if sufficient:
            reason = "Retrieved evidence is sufficient for top-case comparison."
        elif len(retrieved_cases) < self.min_cases:
            reason = "Fewer than three candidate cases were retrieved."
        else:
            reason = "The best semantic match is below the sufficiency threshold."

        return {
            "sufficient": sufficient,
            "reason": reason,
            "best_score": best_score,
        }


@dataclass(frozen=True)
class AnalyzerAgent:
    """Filters retrieved cases and prepares structured evidence for the LLM."""

    top_n: int = 3

    def run(self, query: str, retrieved_cases: List[RetrievedCase]) -> dict:
        top_cases = retrieved_cases[: self.top_n]
        return {
            "query": query,
            "top_cases": top_cases,
            "case_count": len(top_cases),
            "best_case": top_cases[0] if top_cases else None,
            "score_range": self._score_range(top_cases),
        }

    @staticmethod
    def _score_range(cases: List[RetrievedCase]) -> str:
        if not cases:
            return "N/A"
        scores = [score for _, score in cases]
        return f"{min(scores):.3f} - {max(scores):.3f}"
