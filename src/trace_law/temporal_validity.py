"""Temporal-statutory validity layer (TRACE-Law).

India's 2023-24 overhaul replaced three colonial-era codes, in force from
1 July 2024:
    * Indian Penal Code (IPC) 1860            -> Bharatiya Nyaya Sanhita (BNS) 2023
    * Code of Criminal Procedure (CrPC) 1973  -> Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023
    * Indian Evidence Act (IEA) 1872          -> Bharatiya Sakshya Adhiniyam (BSA) 2023

A retriever that surfaces an IPC provision for a present-day query can return
legally stale authority. This layer:
  1. detects statute references (old codes + section numbers) in retrieved text,
  2. maps known sections to their new-code equivalents,
  3. emits human-readable flags so the UI / generator can warn the user.

Rule-based and dependency-free -> fully unit-testable offline. The mapping is a
curated, frequently-litigated subset (not the full 358-section BNS); unknown
sections are still flagged at the code level with a generic correspondence note.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

TRANSITION_DATE = date(2024, 7, 1)

# --- curated section-level correspondences (old -> (new_code, new_section, gloss)) ---
IPC_TO_BNS: Dict[str, str] = {
    "34": "3(5)", "120B": "61", "149": "190", "302": "103", "304B": "80",
    "307": "109", "323": "115", "324": "118", "354": "74", "375": "63",
    "376": "64", "379": "303", "392": "309", "406": "316", "411": "317",
    "415": "318", "420": "318(4)", "498A": "85", "499": "356", "124A": "152",
    "153A": "196", "295A": "299", "304A": "106", "326": "118(2)", "363": "137",
    "366": "87", "201": "238", "467": "338", "468": "336(3)", "471": "340(2)",
}
CRPC_TO_BNSS: Dict[str, str] = {
    "41": "35", "154": "173", "161": "180", "164": "183", "173": "193",
    "190": "210", "200": "223", "204": "227", "313": "351", "320": "359",
    "354": "393", "374": "415", "378": "419", "397": "438", "401": "442",
    "437": "480", "438": "482", "439": "483", "482": "528", "125": "144",
}
IEA_TO_BSA: Dict[str, str] = {
    "3": "2", "8": "6", "24": "22", "25": "23(1)", "27": "23(2)",
    "32": "26", "45": "39", "65A": "62", "65B": "63", "101": "104",
    "106": "109", "114": "119", "118": "124", "133": "138",
}

_CODE_TABLES = {
    "IPC": ("Bharatiya Nyaya Sanhita, 2023 (BNS)", IPC_TO_BNS),
    "CRPC": ("Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS)", CRPC_TO_BNSS),
    "IEA": ("Bharatiya Sakshya Adhiniyam, 2023 (BSA)", IEA_TO_BSA),
}

# how each old code can appear in judgment text
_CODE_PATTERNS = {
    "IPC": r"(?:I\.?P\.?C\.?|Indian Penal Code|Penal Code,?\s*1860)",
    "CRPC": r"(?:Cr\.?\s*P\.?C\.?|Code of Criminal Procedure|Criminal Procedure Code)",
    "IEA": r"(?:Indian Evidence Act|Evidence Act,?\s*1872)",
}
# "Section 302 IPC", "u/s 420 of the IPC", "S. 138", "section 65B of the Evidence Act"
_SECTION_NEAR = re.compile(
    r"(?:section|sec\.?|s\.?|u/?s\.?)\s*([0-9]+[A-Za-z]*(?:\(\d+\))?)",
    re.IGNORECASE,
)


@dataclass
class StatuteFlag:
    code: str                 # IPC / CRPC / IEA
    section: Optional[str]    # e.g. "302" or None if only the code was named
    new_code: str             # BNS / BNSS / BSA full name
    new_section: Optional[str]
    message: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class ValidityReport:
    flags: List[StatuteFlag] = field(default_factory=list)
    stale: bool = False           # any pre-transition code detected
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "stale": self.stale,
            "flags": [f.as_dict() for f in self.flags],
            "notes": self.notes,
        }


def _find_sections_near_code(text: str, code_match: re.Match) -> List[str]:
    """Collect section numbers within a +/- 60 char window of a code mention.

    Retained for backwards compatibility / tests; the main scanner now uses
    nearest-code attribution (see check_text) to avoid cross-code false flags.
    """
    start, end = code_match.span()
    window = text[max(0, start - 60): min(len(text), end + 60)]
    return [m.group(1) for m in _SECTION_NEAR.finditer(window)]


def check_text(text: str, max_dist: int = 45) -> ValidityReport:
    """Scan one document/answer for superseded statute references.

    Each detected section is attributed to the *nearest* code mention (within
    `max_dist` characters). This prevents attaching, e.g., "420" to both IPC and
    CrPC when both codes appear in the same sentence.
    """
    report = ValidityReport()
    if not text:
        return report

    # 1) locate every old-code mention with its position and code key
    code_hits = []  # (center_pos, code)
    for code, pattern in _CODE_PATTERNS.items():
        for m in re.finditer(pattern, text, re.IGNORECASE):
            report.stale = True
            code_hits.append(((m.start() + m.end()) / 2.0, code, m.start(), m.end()))
    if not code_hits:
        return report

    # 2) attribute each section mention to the nearest code mention
    used_codes = set()
    seen = set()
    for sm in _SECTION_NEAR.finditer(text):
        sec = sm.group(1)
        sec_center = (sm.start() + sm.end()) / 2.0
        # distance to a code = gap between the section span and the code span
        best = None
        for center, code, cstart, cend in code_hits:
            if sm.end() <= cstart:
                gap = cstart - sm.end()
            elif cend <= sm.start():
                gap = sm.start() - cend
            else:
                gap = 0
            if best is None or gap < best[0]:
                best = (gap, code, center)
        if best is None or best[0] > max_dist:
            continue  # section not clearly tied to an old code
        _gap, code, center = best
        used_codes.add(center)
        new_code, table = _CODE_TABLES[code]
        norm = sec.replace(" ", "")
        key = (code, norm)
        if key in seen:
            continue
        seen.add(key)
        new_sec = table.get(norm)
        if new_sec:
            msg = (
                f"{code} s.{sec} corresponds to {new_code.split(' (')[0]} "
                f"s.{new_sec}. Offences on/after 1 Jul 2024 are charged under the new code."
            )
        else:
            msg = (
                f"{code} s.{sec} predates the 1 Jul 2024 transition to "
                f"{new_code}; verify the corresponding new-code provision."
            )
        report.flags.append(StatuteFlag(code, sec, new_code, new_sec, msg))

    # 3) any code mentioned without a nearby section -> code-level flag
    for center, code, cstart, cend in code_hits:
        if center in used_codes:
            continue
        new_code, _table = _CODE_TABLES[code]
        if (code, None) in seen:
            continue
        seen.add((code, None))
        report.flags.append(StatuteFlag(
            code, None, new_code, None,
            f"Reference to {code} detected; {new_code} now governs offences "
            f"committed on/after 1 Jul 2024.",
        ))

    if report.stale:
        report.notes.append(
            "Retrieved authority cites a pre-2024 code. It remains valid for acts "
            "committed before 1 Jul 2024 but may be superseded for later conduct."
        )
    return report


def check_documents(docs: List[str]) -> List[ValidityReport]:
    return [check_text(d) for d in docs]


def annotate_query_decision(query: str, decided: Optional[date]) -> Optional[str]:
    """Optional helper: warn if a judgment predates the transition for a query
    that concerns present-day conduct."""
    if decided is not None and decided < TRANSITION_DATE:
        return (
            f"This authority was decided on {decided.isoformat()}, before the "
            "1 Jul 2024 statutory transition."
        )
    return None
