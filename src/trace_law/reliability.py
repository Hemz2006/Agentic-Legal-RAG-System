"""Reliability calibration and abstention (TRACE-Law).

Combines the signals produced upstream into a single [0,1] reliability score and
decides whether the system should answer or abstain. Transparent, monotone, and
tunable -- no black-box calibration so behaviour is explainable and testable.

Signals:
  retrieval_score   : best (normalised) similarity of the evidence set
  margin            : gap between top-1 and top-3 evidence (confidence proxy)
  support_rate      : fraction of cited claims verified (from verification.py)
  authority         : mean authority of the evidence (from authority.py)
  temporal_penalty  : down-weight when stale statutes are flagged unaddressed
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

# weights sum to 1.0 over the positive signals; temporal acts as a penalty.
WEIGHTS = {
    "retrieval": 0.35,
    "margin": 0.10,
    "support": 0.40,
    "authority": 0.15,
}


@dataclass
class ReliabilityResult:
    reliability: float
    decision: str  # "answer" or "abstain"
    components: Dict[str, float]
    reasons: list

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def reliability_score(
    retrieval_score: float,
    support_rate: float,
    authority: float = 0.5,
    margin: float = 0.0,
    temporal_stale: bool = False,
    temporal_flagged: bool = False,
    abstain_threshold: float = 0.45,
) -> ReliabilityResult:
    comp = {
        "retrieval": _clip(retrieval_score),
        "margin": _clip(margin),
        "support": _clip(support_rate),
        "authority": _clip(authority),
    }
    base = sum(WEIGHTS[k] * comp[k] for k in WEIGHTS)

    reasons = []
    # Staleness always costs something. Flagging the user reduces, but does not
    # eliminate, the penalty -- a warning is not a resolution of the staleness.
    if temporal_stale and not temporal_flagged:
        penalty = 0.15
        reasons.append("Superseded statute cited and not flagged to the user.")
    elif temporal_stale and temporal_flagged:
        penalty = 0.05
        reasons.append("Superseded statute cited (flagged to the user).")
    else:
        penalty = 0.0
    reliability = _clip(base - penalty)

    if comp["support"] < 0.5:
        reasons.append("Less than half of cited claims are verified by evidence.")
    if comp["retrieval"] < 0.3:
        reasons.append("Weak retrieval similarity for the evidence set.")

    decision = "answer" if reliability >= abstain_threshold else "abstain"
    if decision == "abstain" and not reasons:
        reasons.append("Combined reliability below the abstention threshold.")
    return ReliabilityResult(reliability, decision, comp, reasons)


def abstention_message(result: ReliabilityResult) -> Optional[str]:
    if result.decision != "abstain":
        return None
    bullet = "\n".join(f"- {r}" for r in result.reasons)
    return (
        "The system is abstaining from a confident answer for this query "
        f"(reliability {result.reliability:.2f}).\n{bullet}\n"
        "Please consult primary sources or a qualified advocate."
    )
