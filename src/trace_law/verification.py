"""Structural citation verification (TRACE-Law).

For every sentence the generator produces that cites a source ([Judgment k]),
check whether the cited evidence actually *entails* the claim. The principled
tool is a natural-language-inference (NLI) model; offline we fall back to a
lexical-entailment heuristic so the layer is always runnable and testable.

A claim is 'supported' if entailment(evidence, claim) is high enough;
'contradicted' if contradiction dominates; otherwise 'unsupported'.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_CITE = re.compile(r"\[(?:judgment|case|source)\s*(\d+)\]", re.IGNORECASE)
_TOKEN = re.compile(r"[a-z0-9]+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_STOP = {
    "the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "was", "were",
    "that", "this", "for", "on", "by", "with", "as", "it", "be", "which", "from",
    "at", "has", "have", "had", "but", "their", "its", "they",
}
_NEG = {"not", "no", "never", "without", "neither", "nor", "cannot", "denied", "rejected"}


def _content_toks(s: str) -> set:
    return {t for t in _TOKEN.findall(s.lower()) if t not in _STOP and len(t) > 2}


def _has_negation(s: str) -> bool:
    toks = set(_TOKEN.findall(s.lower()))
    return bool(toks & _NEG) or "n't" in s.lower()


def split_claims(answer: str) -> List[Tuple[str, List[int]]]:
    """Split an answer into sentences, returning (sentence, [cited_indices])."""
    claims = []
    for sent in _SENT_SPLIT.split(answer.strip()):
        sent = sent.strip()
        if not sent:
            continue
        cites = [int(m.group(1)) for m in _CITE.finditer(sent)]
        claims.append((sent, cites))
    return claims


def lexical_entailment(evidence: str, claim: str) -> float:
    """Offline NLI stand-in: fraction of claim content tokens present in evidence,
    damped when the claim and evidence disagree on polarity (negation mismatch).

    This is a heuristic; a real NLI model (see load_nli) is required for results.
    """
    c = _content_toks(claim)
    if not c:
        return 1.0  # nothing substantive to verify
    e = _content_toks(evidence)
    overlap = len(c & e) / len(c)
    # polarity mismatch: one side negates, the other does not -> likely contradiction
    if _has_negation(claim) != _has_negation(evidence):
        overlap *= 0.4
    return overlap


def load_nli(model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
    """Try to load an NLI model. Returns entail_fn(premise, hypothesis)->float or None."""
    try:  # pragma: no cover - needs model weights
        from transformers import pipeline

        clf = pipeline("text-classification", model=model_name, top_k=None)

        def _entail(premise: str, hypothesis: str) -> float:
            out = clf({"text": premise, "text_pair": hypothesis})
            scores = {d["label"].lower(): d["score"] for d in (out if isinstance(out, list) else [out])}
            return float(scores.get("entailment", 0.0))

        return _entail
    except Exception:
        return None


@dataclass
class ClaimVerdict:
    claim: str
    cited: List[int]
    support: float
    label: str  # supported / unsupported / uncited

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def verify_answer(
    answer: str,
    sources: Sequence[str],
    entail_fn: Optional[Callable[[str, str], float]] = None,
    support_threshold: float = 0.5,
) -> Dict[str, object]:
    """Verify each cited claim against its sources.

    sources: ordered list; [Judgment 1] -> sources[0], etc.
    entail_fn(evidence, claim) -> entailment prob in [0,1]; offline fallback if None.
    """
    fn = entail_fn or lexical_entailment
    verdicts: List[ClaimVerdict] = []
    for sent, cites in split_claims(answer):
        if not cites:
            # Only audit substantive, citation-bearing sentences.
            if _content_toks(sent):
                verdicts.append(ClaimVerdict(sent, [], 0.0, "uncited"))
            continue
        best = 0.0
        for idx in cites:
            if 1 <= idx <= len(sources):
                best = max(best, fn(sources[idx - 1], sent))
        label = "supported" if best >= support_threshold else "unsupported"
        verdicts.append(ClaimVerdict(sent, cites, best, label))

    cited = [v for v in verdicts if v.cited]
    supported = sum(1 for v in cited if v.label == "supported")
    support_rate = supported / len(cited) if cited else 1.0
    return {
        "verdicts": [v.as_dict() for v in verdicts],
        "support_rate": support_rate,
        "num_cited_claims": len(cited),
        "num_supported": supported,
        "num_unsupported": len(cited) - supported,
    }
