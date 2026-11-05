"""The evidence checker: a note may only contain numbers that appear in its case.

Every number in the note (counts, sizes, ratios, prices, IDs, event IDs) is compared with the
numbers in the case. A number is supported if it equals a case value, written in any of the
usual ways: rounded to 0-4 decimals, as a percentage of a ratio, or with thousands commas.
Anything else is reported as an unsupported claim, and the note is rejected.
"""

from __future__ import annotations

import re

from service.schemas import Case, LLMNote

# Notes are shown to reviewers; a link or markup there could only come from injected text.
_LINK_OR_HTML = re.compile(
    r"https?://|www\.|<[a-zA-Z/!]"          # web addresses and HTML tags
    r"|\]\s*[(:]|//"                          # markdown links/images, //host
    r"|(?:javascript|data|mailto|ftp|file):",  # other link schemes
    re.IGNORECASE)
# Any run of digits counts, even next to letters ("trader99999", "x3000").
_NUMBER = re.compile(r"(?<![\d.,])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\d.,])-?\d+(?:\.\d+)?")


def _case_numbers(case: Case) -> list[float]:
    values = [case.alert_id, case.trader_id, case.symbol_id, case.window_seconds,
              *case.event_ids, *case.stats.values()]
    if case.model_score is not None:
        values.append(case.model_score)
    if case.event_ids:
        values.append(len(case.event_ids))
    return [float(v) for v in values]


def _supported(x: float, allowed: list[float]) -> bool:
    for v in allowed:
        for candidate in (v, v * 100.0, abs(v), abs(v) * 100.0):  # ratios may appear as %
            for decimals in range(5):
                if abs(round(candidate, decimals) - x) < 1e-9:
                    return True
    return False


def numbers_in(text: str) -> list[str]:
    return _NUMBER.findall(text)


def check_note(note: LLMNote, case: Case) -> list[str]:
    """Problems with the note; an empty list means every number is backed by the case."""
    allowed = _case_numbers(case)
    problems = []
    for field, text in (("summary", note.summary), ("next_check", note.next_check),
                        *((f"evidence[{i}]", t) for i, t in enumerate(note.evidence))):
        if _LINK_OR_HTML.search(text):
            problems.append(f"{field}: contains a link or HTML, which a note must not")
        for token in numbers_in(text):
            if not _supported(float(token.replace(",", "")), allowed):
                problems.append(f"{field}: number {token} is not in the alert data")
    return problems
