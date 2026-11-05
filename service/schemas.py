"""Data shapes for the case note service, validated with Pydantic."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Pattern = Literal["spoofing", "wash_trade", "pump_and_dump"]


class Case(BaseModel):
    """Everything the note may talk about: one alert, its statistics and its events."""

    model_config = ConfigDict(extra="forbid")

    alert_id: int = Field(ge=0)
    pattern: Pattern
    trader_id: int = Field(ge=0)
    symbol_id: int = Field(ge=0)
    source: Literal["rules", "model"] = "rules"
    model_score: float | None = None
    window_seconds: Literal[1, 60]
    window_start_ns: int = Field(ge=0)
    window_end_ns: int = Field(ge=0)
    # Names are plain identifiers (e.g. large_cancels) so free text cannot reach the LLM here.
    stats: dict[Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,39}$")], float] = (
        Field(max_length=40))
    event_ids: list[int] = Field(max_length=50)  # rows in the events file, oldest first
    event_ids_truncated: bool = False


class LLMNote(BaseModel):
    """The fixed shape the LLM must return. Also used for template notes."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=600)
    evidence: list[str] = Field(min_length=1, max_length=8)
    next_check: str = Field(min_length=1, max_length=300)


class NoteResult(BaseModel):
    alert_id: int
    note: LLMNote
    source: Literal["llm", "template"]
    model: str | None = None  # which LLM wrote it, if any
    checks_passed: bool = True
    rejected_reasons: list[str] = []  # why an LLM note was not used, if it wasn't
