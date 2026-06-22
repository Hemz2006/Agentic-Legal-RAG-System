"""Agent helpers for the LegalAssist workflow (now RRF-based, paper-aligned)."""
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

from config import DECISION_MIN_CASES, DECISION_TAU, RRF_K

RetrievedCase = Tuple[str, float]


@dataclass(frozen=True)
class QueryExpansionAgent:
    """Creates retrieval-focused variants of the user's legal query."""

    def run(self, query: str) -> List[str]:
        q = query.strip()
        if not q:
            return []
        return [
            q,
            f"Supreme Court judgment on {q}",
            f"Indian case law related to {q}",
            f"judgment reasoning provisions sections for {q}",
        ]


@dataclass(frozen=True)
class RetrievalAgent:
    """Runs retrieval for each expanded query and fuses the lists.

    Fusion defaults to Reciprocal Rank Fusion (RRF) -- the principled method --
    instead of the original max-cosine merge (still available as fusion='max').
    """

    retrieve_fn: Callable
    fusion: str = "rrf"
    rrf_k: int = RRF_K

    def run(self, queries: List[str], index, texts: List[str], top_k: int) -> List[RetrievedCase]:
        result_lists = [
            list(self.retrieve_fn(query, index, texts, top_k=top_k)) for query in queries
        ]
        from trace_law.fusion import max_merge, reciprocal_rank_fusion
        if self.fusion == "max":
            return max_merge(result_lists)
        return reciprocal_rank_fusion(result_lists, k=self.rrf_k)


@dataclass(frozen=True)
class DecisionAgent:
    """Checks whether retrieved evidence is strong enough for legal analysis."""

    min_cases: int = DECISION_MIN_CASES
    min_best_score: float = DECISION_TAU  # paper main threshold (0.35)

    def run(self, retrieved_cases: List[RetrievedCase]) -> dict:
        if not retrieved_cases:
            return {"sufficient": False, "reason": "No similar cases were retrieved.", "best_score": 0.0}
        best_score = retrieved_cases[0][1]
        sufficient = len(retrieved_cases) >= self.min_cases and best_score >= self.min_best_score
        if sufficient:
            reason = "Retrieved evidence is sufficient for top-case comparison."
        elif len(retrieved_cases) < self.min_cases:
            reason = f"Fewer than {self.min_cases} candidate cases were retrieved."
        else:
            reason = "The best match is below the sufficiency threshold."
        return {"sufficient": sufficient, "reason": reason, "best_score": best_score}


@dataclass(frozen=True)
class AnalyzerAgent:
    """Filters retrieved cases and prepares structured evidence for the generator."""

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
        scores = [s for _, s in cases]
        return f"{min(scores):.3f} - {max(scores):.3f}"
