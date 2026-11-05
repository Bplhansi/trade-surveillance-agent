"""Runs the Streamlit page headless: it renders, and a click stores a decision in SQLite."""

import json
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from dashboard.prepare import build_review_rows
from dashboard.store import DecisionStore
from service.notes import NoteService
from sim.event_array import make_events, write_event_array

APP = str(Path(__file__).resolve().parents[1] / "app.py")
SEC = 1_000_000_000
T0 = 1_767_605_400 * SEC


def setup_data(tmp_path, monkeypatch):
    ts = T0 + np.array([0, 4, 5, 6, 120]) * SEC
    events = make_events(timestamp_ns=ts, order_id=[1, 2, 3, 4, 5], trader_id=[9, 42, 42, 9, 9],
                         symbol_id=3, price=[10.0, 10.1, 10.2, 10.3, 10.4], quantity=100,
                         event_type=[2, 0, 1, 2, 2], side=0)
    write_event_array(tmp_path / "demo.events.bin", events)
    alerts = [{"alert_id": i, "rule": "spoofing", "timestamp_ns": T0 + 5 * SEC, "event_index": 2,
               "trader_id": 42, "symbol_id": 3, "window": "1s", "source": "rules",
               "model_score": score, "evidence": {"large_cancels": 4, "large_cancel_ratio": 1.0}}
              for i, score in ((1, 7.0), (2, 9.0))]
    rows = build_review_rows(alerts, events, NoteService(None))
    (tmp_path / "demo.review.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "demo.engine.json").write_text(json.dumps({"latency_p50_ns": 1000,
                                                           "latency_p99_ns": 70000}))
    monkeypatch.setenv("TSA_DATA", str(tmp_path))
    monkeypatch.setenv("TSA_NAME", "demo")
    monkeypatch.setenv("TSA_DB", str(tmp_path / "decisions.sqlite"))


def test_dashboard_renders_and_records_a_decision(tmp_path, monkeypatch) -> None:
    setup_data(tmp_path, monkeypatch)
    at = AppTest.from_file(APP, default_timeout=30).run()

    assert not at.exception
    assert "Alert 2" in at.subheader[1].value  # highest score is shown first
    assert at.metric[3].value == "70,000 ns"

    next(b for b in at.button if b.label == "Dismiss").click().run()

    assert not at.exception
    assert DecisionStore(tmp_path / "decisions.sqlite").latest() == {2: "dismiss"}
    assert at.metric[1].value == "100%"  # false alarm rate after one dismissal


def test_dashboard_explains_missing_data(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TSA_DATA", str(tmp_path))
    monkeypatch.setenv("TSA_NAME", "nothing")

    at = AppTest.from_file(APP, default_timeout=30).run()

    assert at.error and "dashboard.prepare" in at.error[0].value
