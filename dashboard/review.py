"""Plain functions behind the review dashboard: queue order, price chart data and metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

SEC = 1_000_000_000
_KIND = {0: "new", 1: "cancel", 2: "trade"}


def _score(row: dict) -> float:
    s = row.get("model_score", row.get("alert", {}).get("model_score"))
    return -math.inf if s is None else float(s)


def queue_order(rows: list[dict]) -> list[dict]:
    """Highest score first; alerts without a score last; ties by alert number."""
    return sorted(rows, key=lambda r: (-_score(r), r["alert_id"]))


def price_window(events: np.ndarray, symbol_id: int, center_ns: int, half_width_s: float,
                 flagged_ids: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trades for the symbol around the alert, and the alert's own events to mark on the chart."""
    ts = events["timestamp_ns"]
    lo = np.searchsorted(ts, center_ns - int(half_width_s * SEC), "left")
    hi = np.searchsorted(ts, center_ns + int(half_width_s * SEC), "right")
    window = events[lo:hi]
    is_trade = (window["symbol_id"] == symbol_id) & (window["type"] == 2)
    trades = pd.DataFrame({
        "time": pd.to_datetime(window["timestamp_ns"][is_trade].astype("int64"), utc=True),
        "price": window["price"][is_trade],
    })
    ids = np.asarray([i for i in flagged_ids if 0 <= i < len(events)], dtype=np.int64)
    marked = events[ids]
    flagged = pd.DataFrame({
        "event_id": ids,
        "time": pd.to_datetime(marked["timestamp_ns"].astype("int64"), utc=True),
        "price": marked["price"],
        "quantity": marked["quantity"],
        "kind": [_KIND.get(int(t), "other") for t in marked["type"]],
    })
    return trades, flagged


def alerts_per_minute(n_alerts: int, events: np.ndarray) -> float:
    """Alerts per minute of simulated market time covered by the events file."""
    if len(events) < 2:
        return 0.0
    minutes = (int(events["timestamp_ns"][-1]) - int(events["timestamp_ns"][0])) / (60 * SEC)
    return n_alerts / minutes if minutes > 0 else 0.0


def false_alarm_rate(decisions: dict[int, str]) -> float | None:
    """Share of decided alerts the reviewer dismissed; None before any decision."""
    if not decisions:
        return None
    return sum(d == "dismiss" for d in decisions.values()) / len(decisions)
