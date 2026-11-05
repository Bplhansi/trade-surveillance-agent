"""Tests for the review dashboard's logic. The Streamlit page only displays these."""

import numpy as np
import pytest

from dashboard.prepare import build_review_rows, parse_engine_summary
from dashboard.review import false_alarm_rate, alerts_per_minute, price_window, queue_order
from dashboard.store import DecisionStore
from service.notes import NoteService
from sim.event_array import make_events

SEC = 1_000_000_000
T0 = 1_767_605_400 * SEC


def alert(alert_id: int, score: float | None, **kw) -> dict:
    row = {"alert_id": alert_id, "rule": "spoofing", "timestamp_ns": T0 + 5 * SEC,
           "event_index": 2, "trader_id": 42, "symbol_id": 3, "window": "1s",
           "source": "rules", "model_score": score, "evidence": {"large_cancels": 4}}
    row.update(kw)
    return row


def events() -> np.ndarray:
    ts = T0 + np.array([0, 4, 5, 6, 120]) * SEC
    return make_events(timestamp_ns=ts, order_id=[1, 2, 3, 4, 5], trader_id=[9, 42, 42, 9, 9],
                       symbol_id=[3, 3, 3, 3, 3], price=[10.0, 10.1, 10.2, 10.3, 10.4],
                       quantity=100, event_type=[2, 0, 1, 2, 2], side=0)


# --- queue -------------------------------------------------------------------------------

def test_queue_is_sorted_by_score_highest_first_missing_scores_last() -> None:
    rows = [alert(1, 7.0), alert(2, None), alert(3, 9.5), alert(4, 7.0)]

    assert [r["alert_id"] for r in queue_order(rows)] == [3, 1, 4, 2]


# --- price chart -------------------------------------------------------------------------

def test_price_window_has_trades_and_marks_the_flagged_events() -> None:
    trades, flagged = price_window(events(), symbol_id=3, center_ns=T0 + 5 * SEC,
                                   half_width_s=10, flagged_ids=[1, 2])

    assert list(trades["price"]) == [10.0, 10.3]  # trades only, inside +/-10 s (not the 120 s one)
    assert list(flagged["event_id"]) == [1, 2]
    assert list(flagged["kind"]) == ["new", "cancel"]
    assert list(flagged["price"]) == [10.1, 10.2]


# --- metrics -----------------------------------------------------------------------------

def test_alerts_per_minute_uses_simulated_market_time() -> None:
    assert alerts_per_minute(n_alerts=6, events=events()) == pytest.approx(3.0)  # 120 s span


def test_false_alarm_rate_comes_from_reviewer_decisions() -> None:
    assert false_alarm_rate({}) is None
    assert false_alarm_rate({1: "accept", 2: "dismiss", 3: "dismiss", 4: "accept"}) == 0.5


# --- decisions (SQLite) --------------------------------------------------------------------

def test_decisions_are_stored_with_a_timestamp_and_latest_wins(tmp_path) -> None:
    store = DecisionStore(tmp_path / "decisions.sqlite")
    store.record(7, "dismiss", comment="normal market making")
    store.record(7, "accept")
    store.record(8, "dismiss")

    again = DecisionStore(tmp_path / "decisions.sqlite")  # survives a restart
    assert again.latest() == {7: "accept", 8: "dismiss"}
    history = again.history(7)
    assert [h["decision"] for h in history] == ["dismiss", "accept"]
    assert history[0]["comment"] == "normal market making"
    assert history[0]["decided_at"].endswith("+00:00")  # UTC


def test_only_accept_or_dismiss_can_be_stored(tmp_path) -> None:
    store = DecisionStore(tmp_path / "d.sqlite")

    with pytest.raises(ValueError):
        store.record(1, "maybe")


# --- prepare -----------------------------------------------------------------------------

SUMMARY = """
summary
events                251,805
throughput             85,939 events/s
latency mean           11,511 ns
latency p50             1,100 ns
latency p95            55,300 ns
latency p99           103,300 ns
latency max         1,834,600 ns
alerts                     85
"""


def test_engine_summary_is_parsed() -> None:
    stats = parse_engine_summary(SUMMARY)

    assert stats == {"events": 251805, "throughput_events_per_s": 85939,
                     "latency_mean_ns": 11511, "latency_p50_ns": 1100, "latency_p95_ns": 55300,
                     "latency_p99_ns": 103300, "latency_max_ns": 1834600, "alerts": 85}


def test_engine_summary_missing_is_an_error() -> None:
    with pytest.raises(ValueError):
        parse_engine_summary("no summary here")


def test_every_alert_gets_a_review_row_with_a_checked_note() -> None:
    rows = build_review_rows([alert(1, 7.0), alert(2, None)], events(), NoteService(None))

    assert [r["alert_id"] for r in rows] == [1, 2]
    assert all(r["note"]["checks_passed"] for r in rows)
    assert rows[0]["case"]["event_ids"] == [1, 2]
