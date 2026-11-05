"""Tests for loading features/labels and turning model scores into alerts."""

import json
from pathlib import Path

import numpy as np
import pytest

from ml.dataset import CLASSES, load_features, load_labels
from ml.evaluate_alerts import Alert, Case, score
from ml.model_alerts import EventKeys, alerts_from_scores, choose_threshold

SEC = 1_000_000_000


def keys(n: int, trader: int = 7, symbol: int = 1, step: int = SEC) -> EventKeys:
    return EventKeys(
        trader_id=np.full(n, trader, dtype=np.uint32),
        symbol_id=np.full(n, symbol, dtype=np.uint32),
        timestamp_ns=np.arange(n, dtype=np.uint64) * step,
    )


# --- dataset -------------------------------------------------------------------------

def test_load_features_reads_rows_and_column_names(tmp_path: Path) -> None:
    rows = np.arange(6, dtype=np.float32).reshape(2, 3)
    rows.tofile(tmp_path / "x.features.bin")
    (tmp_path / "x.features.json").write_text(
        json.dumps({"rows": 2, "dtype": "float32", "columns": ["a", "b", "c"]}))

    X, columns = load_features(tmp_path, "x")

    assert columns == ["a", "b", "c"]
    np.testing.assert_array_equal(X, rows)


def test_load_features_rejects_a_size_mismatch(tmp_path: Path) -> None:
    np.zeros(5, dtype=np.float32).tofile(tmp_path / "x.features.bin")
    (tmp_path / "x.features.json").write_text(
        json.dumps({"rows": 2, "dtype": "float32", "columns": ["a", "b", "c"]}))

    with pytest.raises(ValueError, match="rows"):
        load_features(tmp_path, "x")


def test_load_labels_marks_pattern_events_by_class(tmp_path: Path) -> None:
    (tmp_path / "x.labels.csv").write_text(
        "event_index,order_id,trader_id,pattern,pattern_id\n"
        "1,5,7,spoofing,0\n3,6,8,pump_and_dump,1\n")

    y = load_labels(tmp_path, "x", n_events=5)

    assert CLASSES == ("none", "spoofing", "wash_trade", "pump_and_dump")
    np.testing.assert_array_equal(y, [0, 1, 0, 3, 0])


# --- alerts from scores ---------------------------------------------------------------

def test_events_at_or_above_the_threshold_become_alerts() -> None:
    scores = np.array([0.1, 0.9, 0.5, 0.95])
    rules = np.array(["spoofing"] * 4)

    alerts = alerts_from_scores(scores, rules, keys(4, step=100 * SEC), threshold=0.5, cooldown_ns=0)

    assert [a.timestamp_ns for a in alerts] == [100 * SEC, 200 * SEC, 300 * SEC]
    assert all(a.rule == "spoofing" and a.trader_id == 7 for a in alerts)


def test_cooldown_matches_the_engine_one_alert_per_rule_trader_symbol() -> None:
    scores = np.ones(5)
    rules = np.array(["wash_trade"] * 5)

    alerts = alerts_from_scores(scores, rules, keys(5, step=30 * SEC), threshold=0.5,
                                cooldown_ns=60 * SEC)

    # t = 0, 30, 60, 90, 120 s -> alerts at 0 and 60 and 120
    assert [a.timestamp_ns // SEC for a in alerts] == [0, 60, 120]


def test_choose_threshold_picks_the_best_f1_on_the_given_cases() -> None:
    k = keys(4, step=100 * SEC)
    cases = [Case(0, "spoofing", 7, 1, 300 * SEC, 300 * SEC)]
    scores = np.array([0.2, 0.6, 0.3, 0.9])  # only the last event is inside the case
    rules = np.array(["spoofing"] * 4)

    threshold, f1 = choose_threshold(scores, rules, k, cases, n_events=4,
                                     candidates=[0.1, 0.5, 0.8], cooldown_ns=0)

    assert threshold == 0.8
    assert f1 == 1.0


# --- pattern-agnostic "anomaly" alerts (Isolation Forest) ------------------------------

def test_anomaly_alert_matches_a_case_of_any_pattern() -> None:
    case = Case(0, "wash_trade", 7, 1, 100 * SEC, 110 * SEC)

    result = score([Alert("anomaly", 7, 1, 105 * SEC), Alert("anomaly", 9, 1, 105 * SEC)],
                   [case], n_events=10_000)

    assert result["wash_trade"].recall == 1.0
    assert result["overall"].alerts == 2
    assert result["overall"].true_positives == 1
    assert result["overall"].precision == 0.5


def test_choose_threshold_takes_the_middle_of_tied_best_thresholds() -> None:
    k = keys(4, step=100 * SEC)
    cases = [Case(0, "spoofing", 7, 1, 300 * SEC, 300 * SEC)]
    scores = np.array([0.2, 0.6, 0.3, 0.9])
    rules = np.array(["spoofing"] * 4)

    # 0.7, 0.8 and 0.85 all give F1 = 1.0; the middle one is the least fragile choice.
    threshold, _ = choose_threshold(scores, rules, k, cases, n_events=4,
                                    candidates=[0.1, 0.7, 0.8, 0.85], cooldown_ns=0)

    assert threshold == 0.8
