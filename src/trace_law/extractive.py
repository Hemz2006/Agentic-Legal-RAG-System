"""Offline extractive FIRAC brief builder (consolidated from the old rag_pipeline).

This is the single deterministic generator used when no LLM is configured. It
extracts readable Facts / Rules / Analysis / Conclusion passages from the
retrieved judgments -- no external model, fully testable offline.
"""
from __future__ import annotations

import re
from typing import List, Sequence, Set, Tuple


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
    return cleaned[:limit].rsplit(" ", 1)[0].strip()


def _sentences(text: str) -> List[str]:
    cleaned = _clean_text(text)
    protected = {"No.": "No<dot>", "Nos.": "Nos<dot>", "Mr.": "Mr<dot>", "Mrs.": "Mrs<dot>",
                 "Dr.": "Dr<dot>", "J.": "J<dot>", "Ltd.": "Ltd<dot>", "vs.": "vs<dot>", "v.": "v<dot>"}
    for a, b in protected.items():
        cleaned = cleaned.replace(a, b)
    raw = [p.strip().replace("<dot>", ".") for p in re.split(r"(?<=[.!?])\s+", cleaned) if p.strip()]
    return [s for s in raw if _is_complete_sentence(s)]


def _is_complete_sentence(s: str) -> bool:
    if len(s) < 55 or len(s.split()) < 9:
        return False
    if s.lower().endswith((" no.", " nos.", " v.", " vs.", " mr.", " dr.", "crl.", "appeal no.")):
        return False
    return True


def _query_terms(query: str) -> Set[str]:
    return {t.lower() for t in re.findall(r"[a-zA-Z0-9]+", query) if len(t) > 2}


def _sentence_score(sentence: str, terms: Set[str]) -> int:
    low = sentence.lower()
    score = sum(2 for t in terms if t in low)
    markers = ("section", "article", "ipc", "crpc", "act", "court", "held", "appeal",
               "conviction", "sentence", "petitioner", "respondent")
    return score + sum(1 for m in markers if m in low)


def _important_excerpt(text: str, query: str, limit: int = 520) -> str:
    terms = _query_terms(query)
    ranked = sorted(_sentences(text)[:160], key=lambda s: _sentence_score(s, terms), reverse=True)
    return _clean_snippet(ranked[0] if ranked else " ".join(text.split()), limit=limit)


def _facts(text: str, query: str) -> List[str]:
    terms = _query_terms(query)
    markers = ("facts", "occurrence", "alleged", "accused", "appellant", "respondent",
               "prosecution", "trial", "convicted", "charged", "petition", "appeal")
    cands = [s for s in _sentences(text)[:80] if any(m in s.lower() for m in markers)]
    if not cands:
        cands = [_important_excerpt(text, query, limit=700)]
    cands = sorted(cands, key=lambda s: _sentence_score(s, terms), reverse=True)
    return [_clean_snippet(s, limit=320) for s in cands[:4]]


def _rules(judgments: Sequence[str]) -> str:
    patterns = [
        r"(?:section|s\.|sec\.)\s+\d+[A-Za-z-]*(?:\(\d+\))*",
        r"article\s+\d+[A-Za-z-]*",
        r"Indian Penal Code|IPC|CrPC|BNS|BNSS|BSA|Code of Criminal Procedure|Constitution of India",
        r"[A-Z][A-Za-z ]+ Act,?\s+\d{4}",
    ]
    found: List[str] = []
    for j in judgments:
        for pat in patterns:
            found.extend(re.findall(pat, j, flags=re.IGNORECASE))
    uniq: List[str] = []
    for item in found:
        norm = " ".join(item.split())
        if norm.lower() not in {x.lower() for x in uniq}:
            uniq.append(norm)
    if not uniq:
        return "- Not expressly stated in retrieved text."
    return "\n".join(f"- {x}" for x in uniq[:12])


def _conclusion(text: str) -> str:
    markers = ("appeal is allowed", "appeal allowed", "appeal dismissed", "petition dismissed",
               "petition allowed", "conviction", "sentence", "order")
    for s in reversed(_sentences(text)[-80:]):
        if any(m in s.lower() for m in markers):
            return s[:900]
    return "The final holding is not clearly available from the retrieved excerpt."


def _bullets(items: List[str]) -> str:
    if not items:
        return "- No clear facts were extracted from the retrieved judgment text."
    return "\n".join(f"- {x}" for x in items)


def firac_brief(query: str, retrieved: Sequence[Tuple[str, float]]) -> str:
    """Build a readable FIRAC research brief from retrieved (text, score) pairs."""
    if not retrieved:
        return "Not enough data available in the retrieved documents."
    top = list(retrieved)[:3]
    lines = [
        f"{i+1}. **[Judgment {i+1}]** similarity={s:.3f}. {_clean_snippet(_important_excerpt(d, query, 520))}..."
        for i, (d, s) in enumerate(top)
    ]
    facts = _facts(top[0][0], query)
    rules = _rules([d for d, _ in top])
    analysis = "\n".join(
        f"- **[Judgment {i+1}]** matches the query (similarity {s:.3f}). "
        f"Passage: \"{_clean_snippet(_important_excerpt(d, query, 650))}...\""
        for i, (d, s) in enumerate(top)
    )
    conclusion = _clean_snippet(_conclusion(top[0][0]), limit=900)
    return (
        f"**Top Relevant Judgments:**\n" + "\n".join(lines) + "\n\n"
        f"**F - Facts:**\n{_bullets(facts)}\n\n"
        f"**I - Issues:**\n1. Whether the retrieved judgments address `{query}`.\n"
        "2. Whether the material facts are comparable.\n"
        "3. Whether the cited provisions support the conclusion.\n\n"
        f"**R - Rules:**\n{rules}\n\n"
        f"**A - Analysis:**\n{analysis}\n\n"
        f"**C - Conclusion:**\n{conclusion}\n\n"
        "**Research Note:** Offline extractive brief (no external LLM). Research assistance, not legal advice."
    )
