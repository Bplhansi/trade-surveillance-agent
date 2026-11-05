"""Tests for ml/evaluate_alerts.py: grading engine alerts against the injected cases."""

import json
from pathlib import Path

import pytest

from ml.evaluate_alerts import (
    GRACE_NS,
    PATTERNS,
    Alert,
    Case,
    load_alerts,
    load_cases,
    main,
    score,
)

SEC = 1_000_000_000

SPOOF = Case(pattern_id=0, pattern="spoofing", trader_id=7, symbol_id=1, start_ns=100 * SEC, end_ns=110 * SEC)
WASH = Case(pattern_id=1, pattern="wash_trade", trader_id=8, symbol_id=2, start_ns=200 * SEC, end_ns=260 * SEC)


def alert(rule: str, trader: int, symbol: int, ts: int) -> Alert:
    return Alert(rule=rule, trader_id=trader, symbol_id=symbol, timestamp_ns=ts)


def test_perfect_alerts_give_precision_and_recall_of_one() -> None:
    alerts = [alert("spoofing", 7, 1, 105 * SEC), alert("wash_trade", 8, 2, 230 * SEC)]

    result = score(alerts, [SPOOF, WASH], n_events=10_000)

    for name in ("spoofing", "wash_trade"):
        assert result[name].precision == 1.0
        assert result[name].recall == 1.0
    assert result["overall"].precision == 1.0
    assert result["overall"].recall == 1.0


def test_alert_on_the_wrong_trader_is_a_false_positive() -> None:
    result = score([alert("spoofing", 99, 1, 105 * SEC)], [SPOOF], n_events=10_000)

    assert result["spoofing"].false_positives == 1
    assert result["spoofing"].precision == 0.0
    assert result["spoofing"].recall == 0.0


def test_alert_with_the_wrong_pattern_type_does_not_count() -> None:
    # Right trader and time, but the wash rule fired on a spoofing case.
    result = score([alert("wash_trade", 7, 1, 105 * SEC)], [SPOOF], n_events=10_000)

    assert result["wash_trade"].false_positives == 1
    assert result["spoofing"].recall == 0.0


def test_a_missed_case_lowers_recall() -> None:
    other = Case(pattern_id=2, pattern="spoofing", trader_id=9, symbol_id=3, start_ns=0, end_ns=SEC)

    result = score([alert("spoofing", 7, 1, 105 * SEC)], [SPOOF, other], n_events=10_000)

    assert result["spoofing"].cases == 2
    assert result["spoofing"].caught == 1
    assert result["spoofing"].recall == 0.5
    assert result["spoofing"].precision == 1.0


def test_several_alerts_on_one_case_catch_it_once() -> None:
    alerts = [alert("spoofing", 7, 1, 101 * SEC), alert("spoofing", 7, 1, 109 * SEC)]

    result = score(alerts, [SPOOF], n_events=10_000)

    assert result["spoofing"].caught == 1
    assert result["spoofing"].true_positives == 2
    assert result["spoofing"].precision == 1.0


@pytest.mark.parametrize(
    ("ts", "matches"),
    [
        (SPOOF.end_ns + GRACE_NS, True),        # just inside the grace period after the case
        (SPOOF.end_ns + GRACE_NS + 1, False),   # just outside it
        (SPOOF.start_ns - GRACE_NS, True),
        (SPOOF.start_ns - GRACE_NS - 1, False),
    ],
)
def test_grace_period_around_a_case(ts: int, matches: bool) -> None:
    result = score([alert("spoofing", 7, 1, ts)], [SPOOF], n_events=10_000)

    assert (result["spoofing"].caught == 1) is matches


def test_no_alerts_means_precision_is_undefined_not_zero() -> None:
    result = score([], [SPOOF], n_events=10_000)

    assert result["spoofing"].precision is None
    assert result["spoofing"].recall == 0.0


def test_false_alarms_are_also_reported_per_10k_events() -> None:
    alerts = [alert("spoofing", 99, 1, 0)] * 3

    result = score(alerts, [SPOOF], n_events=20_000)

    assert result["spoofing"].false_alarms_per_10k == pytest.approx(1.5)


def test_every_pattern_gets_a_row_even_without_cases_or_alerts() -> None:
    result = score([], [], n_events=1)

    assert set(result) == {*PATTERNS, "overall"}
    assert result["pump_and_dump"].recall is None  # no cases: recall undefined


def test_files_are_loaded(tmp_path: Path) -> None:
    patterns = tmp_path / "x.patterns.csv"
    patterns.write_text(
        "pattern_id,pattern,trader_id,symbol_id,start_ns,end_ns\n"
        f"0,spoofing,7,1,{SPOOF.start_ns},{SPOOF.end_ns}\n"
    )
    alerts = tmp_path / "x.alerts.jsonl"
    alerts.write_text(
        json.dumps({"alert_id": 1, "rule": "spoofing", "timestamp_ns": 105 * SEC,
                    "event_index": 3, "trader_id": 7, "symbol_id": 1, "window": "1s",
                    "evidence": {}}) + "\n\n"  # a blank line is allowed
    )

    assert load_cases(patterns) == [SPOOF]
    assert load_alerts(alerts) == [alert("spoofing", 7, 1, 105 * SEC)]


def test_unknown_rule_name_in_alerts_is_an_error(tmp_path: Path) -> None:
    alerts = tmp_path / "bad.jsonl"
    alerts.write_text(json.dumps({"rule": "spoofng", "trader_id": 1, "symbol_id": 1,
                                  "timestamp_ns": 0}) + "\n")

    with pytest.raises(ValueError, match="spoofng"):
        load_alerts(alerts)


def test_main_prints_precision_and_recall_per_pattern(tmp_path: Path, capsys) -> None:
    (tmp_path / "v.patterns.csv").write_text(
        "pattern_id,pattern,trader_id,symbol_id,start_ns,end_ns\n"
        f"0,spoofing,7,1,{SPOOF.start_ns},{SPOOF.end_ns}\n"
    )
    (tmp_path / "v.events.bin").write_bytes(b"\0" * 40 * 500)
    alerts = tmp_path / "v.alerts.jsonl"
    alerts.write_text(json.dumps({"rule": "spoofing", "trader_id": 7, "symbol_id": 1,
                                  "timestamp_ns": 105 * SEC}) + "\n")

    code = main(["--data", str(tmp_path), "--name", "v", "--alerts", str(alerts)])

    out = capsys.readouterr().out
    assert code == 0
    assert "500 events" in out
    for name in (*PATTERNS, "overall"):
        assert name in out
    assert "precision" in out and "recall" in out


def test_alerts_can_be_filtered_by_source(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    rows = [
        {"rule": "spoofing", "trader_id": 1, "symbol_id": 1, "timestamp_ns": 1, "source": "rules"},
        {"rule": "spoofing", "trader_id": 2, "symbol_id": 1, "timestamp_ns": 2, "source": "model"},
        {"rule": "spoofing", "trader_id": 3, "symbol_id": 1, "timestamp_ns": 3},  # rules-only file
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    assert [a.trader_id for a in load_alerts(path)] == [1, 2, 3]
    assert [a.trader_id for a in load_alerts(path, source="rules")] == [1, 3]
    assert [a.trader_id for a in load_alerts(path, source="model")] == [2]
