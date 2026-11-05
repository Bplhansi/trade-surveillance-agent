"""Turn per-event model scores into alerts, the same way the engine's rules raise them.

An event becomes an alert when its score is at or above the threshold. Like the engine
(config/rules.ini, [general] cooldown_s), at most one alert per rule + trader + symbol is
raised per cooldown, so models and rules are graded on equal terms.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.evaluate_alerts import Alert, Case, score

COOLDOWN_NS = 60 * 1_000_000_000


@dataclass(frozen=True)
class EventKeys:
    """Who/what/when for each event. Used to build and grade alerts, never as model input."""

    trader_id: np.ndarray
    symbol_id: np.ndarray
    timestamp_ns: np.ndarray

    @classmethod
    def from_events_file(cls, path: Path) -> EventKeys:
        from sim.event_array import read_event_array

        ev = read_event_array(path)
        return cls(ev["trader_id"].copy(), ev["symbol_id"].copy(), ev["timestamp_ns"].copy())


def alerts_from_scores(scores: np.ndarray, rules: np.ndarray, keys: EventKeys, threshold: float,
                       cooldown_ns: int = COOLDOWN_NS) -> list[Alert]:
    """Alerts for events with score >= threshold, in event order, applying the cooldown."""
    last: dict[tuple[str, int, int], int] = {}
    alerts = []
    for i in np.flatnonzero(scores >= threshold):
        rule = str(rules[i])
        trader, symbol, ts = int(keys.trader_id[i]), int(keys.symbol_id[i]), int(keys.timestamp_ns[i])
        key = (rule, trader, symbol)
        if key in last and ts - last[key] < cooldown_ns:
            continue
        last[key] = ts
        alerts.append(Alert(rule=rule, trader_id=trader, symbol_id=symbol, timestamp_ns=ts))
    return alerts


def case_f1(alerts: Sequence[Alert], cases: Sequence[Case], n_events: int) -> float:
    overall = score(alerts, cases, n_events)["overall"]
    p, r = overall.precision or 0.0, overall.recall or 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def choose_threshold(scores: np.ndarray, rules: np.ndarray, keys: EventKeys,
                     cases: Sequence[Case], n_events: int, candidates: Sequence[float],
                     cooldown_ns: int = COOLDOWN_NS) -> tuple[float, float]:
    """The candidate threshold with the best case-level F1.

    When several thresholds tie for the best F1 (common when a model separates the cases
    cleanly), the middle one is taken: the extremes are the most fragile on new data.
    """
    results = [(float(t), case_f1(alerts_from_scores(scores, rules, keys, t, cooldown_ns),
                                  cases, n_events)) for t in sorted(candidates)]
    best_f1 = max(f1 for _, f1 in results)
    tied = [t for t, f1 in results if f1 == best_f1]
    return tied[(len(tied) - 1) // 2], best_f1


def quantile_candidates(scores: np.ndarray, n: int = 40) -> list[float]:
    """Thresholds that flag between ~10% and ~0.001% of events."""
    q = 1.0 - np.logspace(-1, -5, n)
    return sorted({float(v) for v in np.quantile(scores, q)})
