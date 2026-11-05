"""Make one case note: ask the LLM, check it, fall back to a template if anything is off.

The LLM explains an alert; it never decides anything. Its reply must be the fixed JSON shape
(LLMNote), and every number in it must appear in the case (service/validate.py). If the LLM is
unavailable, errors, refuses, returns bad JSON, or makes an unsupported claim, a plain template
note built only from the case is used instead, so every alert always gets a note.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import ValidationError

from service.schemas import Case, LLMNote, NoteResult
from service.validate import check_note

_DESCRIPTIONS = {
    "spoofing": "a burst of large orders that were cancelled shortly after being placed",
    "wash_trade": "buying and selling the same symbol in near-equal amounts, partly with itself",
    "pump_and_dump": "selling after its own buying had pushed the price up",
}

_NEXT_CHECKS = {
    "spoofing": "Check whether the cancelled orders sat at the best price and whether the trader "
                "traded on the other side while they were showing.",
    "wash_trade": "Check whether both sides of the self-trades belong to the same beneficial "
                  "owner and whether the trades changed the trader's position.",
    "pump_and_dump": "Check the trader's position before and after the price rise and any "
                     "related news or messages promoting the symbol.",
}

_KEY_STATS = {
    "spoofing": ("large_orders", "large_cancels", "large_cancel_ratio", "max_order_qty"),
    "wash_trade": ("trades", "buy_qty", "sell_qty", "net_qty", "self_trades"),
    "pump_and_dump": ("buy_qty", "sell_qty", "net_qty", "first_trade_price", "price_change"),
}


class NoteGenerator(Protocol):
    model: str

    def generate(self, case: Case) -> str:
        """Return the LLM's raw reply (expected: LLMNote as JSON)."""
        ...


def _fmt(value: float) -> str:
    """At most 4 decimals, trailing zeros dropped: the precision the checker accepts."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def template_note(case: Case) -> LLMNote:
    """A plain note that only repeats numbers from the case."""
    stats = [f"{k} = {_fmt(case.stats[k])}" for k in _KEY_STATS[case.pattern] if k in case.stats]
    if case.event_ids:
        stats.append(f"{len(case.event_ids)} event(s) of this trader in the window, "
                     f"first {case.event_ids[0]}, last {case.event_ids[-1]}")
    if case.model_score is not None:
        stats.append(f"model score {_fmt(case.model_score)}")
    return LLMNote(
        summary=(f"Trader {case.trader_id} on symbol {case.symbol_id} was flagged for possible "
                 f"{case.pattern.replace('_', ' ')} ({_DESCRIPTIONS[case.pattern]}) by the "
                 f"{case.source}, based on the last {case.window_seconds} second(s) of activity."),
        evidence=stats[:8],
        next_check=_NEXT_CHECKS[case.pattern],
    )


class NoteService:
    def __init__(self, llm: NoteGenerator | None) -> None:
        self.llm = llm

    def make_note(self, case: Case) -> NoteResult:
        reasons: list[str] = []
        if self.llm is None:
            reasons.append("no LLM configured")
        else:
            try:
                note = LLMNote.model_validate_json(self.llm.generate(case))
                reasons = check_note(note, case)
                if not reasons:
                    return NoteResult(alert_id=case.alert_id, note=note, source="llm",
                                      model=getattr(self.llm, "model", None))
            except ValidationError as e:
                reasons.append(f"reply did not match the note format ({e.error_count()} error(s))")
            except Exception as e:  # noqa: BLE001 - any LLM failure must fall back, not crash
                reasons.append(f"LLM call failed: {type(e).__name__}")
        note = template_note(case)
        return NoteResult(alert_id=case.alert_id, note=note, source="template",
                          checks_passed=not check_note(note, case), rejected_reasons=reasons)
