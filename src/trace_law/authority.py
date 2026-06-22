"""Authority-weighted evidence ranking (TRACE-Law).

Pure similarity ignores that, in a common-law system, *who* decided a case and
*how it has been treated* matter. This module derives an authority score from
signals that can be parsed from judgment text/metadata:

  * court level        : Supreme Court > High Court > Tribunal/District
  * bench strength     : Constitution Bench (5+) > 3-judge > Division (2) > Single
  * precedent treatment: 'overruled' down-weights, 'affirmed/followed' up-weights
  * recency            : mild preference for more recent decisions

The final ranking score blends semantic similarity with authority:
    final = (1 - alpha) * sim_norm + alpha * authority
Pure logic -> unit-testable offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Sequence, Tuple

# court level weights
_COURT_PATTERNS = [
    (re.compile(r"\bsupreme court\b", re.I), 1.0, "Supreme Court"),
    (re.compile(r"\bhigh court\b", re.I), 0.7, "High Court"),
    (re.compile(r"\b(tribunal|appellate tribunal|nclat|nclt|itat|cestat)\b", re.I), 0.45, "Tribunal"),
    (re.compile(r"\b(district court|sessions court|magistrate)\b", re.I), 0.3, "District/Sessions"),
]


def _court_level(text: str) -> Tuple[float, str]:
    for pat, w, name in _COURT_PATTERNS:
        if pat.search(text):
            return w, name
    return 0.5, "Unknown"  # neutral prior


def _bench_strength(text: str) -> Tuple[float, str]:
    t = text.lower()
    if re.search(r"constitution bench|five[- ]judge|5[- ]judge|seven[- ]judge|nine[- ]judge", t):
        return 1.0, "Constitution Bench"
    if re.search(r"three[- ]judge|3[- ]judge|full bench", t):
        return 0.8, "Three-Judge/Full Bench"
    if re.search(r"division bench|two[- ]judge|2[- ]judge", t):
        return 0.6, "Division Bench"
    if re.search(r"single (judge|bench)|learned single", t):
        return 0.4, "Single Judge"
    return 0.55, "Unspecified"


_TREATMENT = [
    (re.compile(r"\b(overruled|set aside|reversed|no longer good law)\b", re.I), -0.4, "overruled"),
    (re.compile(r"\b(per incuriam)\b", re.I), -0.3, "per incuriam"),
    (re.compile(r"\b(affirmed|upheld|followed|reiterated|relied upon)\b", re.I), +0.2, "affirmed/followed"),
    (re.compile(r"\b(distinguished)\b", re.I), -0.05, "distinguished"),
]

# A negative treatment word often refers to *another* court's decision being
# overruled by the present (higher) court, which makes the present judgment MORE
# authoritative, not less. If the negative word sits next to a lower-court
# mention, we do not penalise. This addresses the polarity-inversion failure.
_LOWER_COURT_NEAR = re.compile(
    r"\b(high court|trial court|sessions court|district court|tribunal|lower court|"
    r"impugned (?:judgment|order)|appealed (?:judgment|order))\b",
    re.I,
)


def _treatment(text: str) -> Tuple[float, List[str]]:
    delta = 0.0
    labels: List[str] = []
    for pat, d, name in _TREATMENT:
        m = pat.search(text)
        if not m:
            continue
        if d < 0:
            s, e = m.span()
            window = text[max(0, s - 60): min(len(text), e + 60)]
            if _LOWER_COURT_NEAR.search(window):
                # Present judgment overrules a lower one -> do not penalise it.
                labels.append(f"{name} (of lower court)")
                continue
        delta += d
        labels.append(name)
    return delta, labels


_YEAR = re.compile(r"\b(19|20)\d{2}\b")


def _recency(text: str, decided: Optional[date]) -> float:
    year = decided.year if decided else None
    if year is None:
        years = [int(m.group(0)) for m in _YEAR.finditer(text)]
        years = [y for y in years if 1900 <= y <= date.today().year]
        if years:
            year = max(years)
    if year is None:
        return 0.5
    # linear ramp 1950 -> today mapped to 0.3 .. 1.0
    span = max(1, date.today().year - 1950)
    return 0.3 + 0.7 * max(0.0, min(1.0, (year - 1950) / span))


@dataclass
class AuthorityScore:
    authority: float
    court: str
    bench: str
    treatment: List[str]
    recency: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def score_authority(text: str, decided: Optional[date] = None, meta: Optional[dict] = None) -> AuthorityScore:
    """Authority score from text, optionally overridden by trusted metadata.

    `meta` may supply any of: court_level (0-1), bench (0-1), recency (0-1),
    treatment_delta (float), court (str), bench_name (str). Supplying metadata
    avoids the well-known failure modes of text parsing (e.g. "overruled"
    referring to the *lower* court, or recency read from a cited year). Pass
    IL-TUR / court metadata in production; regex is the fallback.
    """
    meta = meta or {}
    court_w, court = _court_level(text)
    bench_w, bench = _bench_strength(text)
    treat_delta, treat_labels = _treatment(text)
    rec = _recency(text, decided)

    if "court_level" in meta:
        court_w = float(meta["court_level"]); court = meta.get("court", court)
    if "bench" in meta:
        bench_w = float(meta["bench"]); bench = meta.get("bench_name", bench)
    if "recency" in meta:
        rec = float(meta["recency"])
    if "treatment_delta" in meta:
        treat_delta = float(meta["treatment_delta"])
        treat_labels = meta.get("treatment", treat_labels)

    raw = 0.45 * court_w + 0.30 * bench_w + 0.25 * rec + treat_delta
    authority = max(0.0, min(1.0, raw))
    return AuthorityScore(authority, court, bench, treat_labels, rec)


def _normalize(scores: Sequence[float]) -> List[float]:
    """Scale scores to ~[0,1] preserving relative magnitude.

    For non-negative scores (cosine, BM25, RRF) we divide by the max, which
    keeps near-equal top scores near-equal -- unlike min-max stretching, which
    over-separates a tiny gap (e.g. 0.81 vs 0.80) on small candidate sets.
    """
    if not scores:
        return []
    hi = max(scores)
    if hi > 0:
        return [max(0.0, s) / hi for s in scores]
    lo = min(scores)  # all <= 0: fall back to min-max
    if hi - lo < 1e-9:
        return [0.5 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def rerank_by_authority(
    results: Sequence[Tuple[str, float]],
    alpha: float = 0.3,
    metas: Optional[Sequence[dict]] = None,
) -> List[Tuple[str, float, dict]]:
    """Blend similarity with authority.

    results: [(doc_text, sim_score), ...]
    metas:   optional per-doc metadata dicts (see score_authority) aligned to results.
    returns: [(doc_text, final_score, detail_dict), ...] sorted best-first.
    """
    if not results:
        return []
    docs = [d for d, _ in results]
    sims = [s for _, s in results]
    sim_norm = _normalize(sims)
    if metas is None:
        metas = [None] * len(docs)
    auth = [score_authority(d, meta=m) for d, m in zip(docs, metas)]
    out = []
    for doc, sim, sn, a in zip(docs, sims, sim_norm, auth):
        final = (1 - alpha) * sn + alpha * a.authority
        detail = a.as_dict()
        detail["similarity"] = sim
        detail["similarity_norm"] = sn
        detail["final"] = final
        out.append((doc, final, detail))
    out.sort(key=lambda x: x[1], reverse=True)
    return out
