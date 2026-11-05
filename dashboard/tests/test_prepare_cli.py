"""Runs the command-line entry points end to end, with a stand-in for the C++ engine."""

import json
import subprocess

import httpx
import numpy as np

from dashboard import prepare
from service.case import main as case_main
from service.notes import NoteService
from service.schemas import Case
from sim.event_array import make_events, write_event_array

SEC = 1_000_000_000
T0 = 1_767_605_400 * SEC
SUMMARY = "summary\nevents  5\nthroughput  100 events/s\nlatency p50  1,000 ns\n" \
          "latency p99  9,000 ns\nalerts  1\n"
ALERT = {"alert_id": 1, "rule": "wash_trade", "timestamp_ns": T0 + 5 * SEC, "event_index": 2,
         "trader_id": 42, "symbol_id": 3, "window": "60s", "source": "rules", "model_score": 8.0,
         "evidence": {"trades": 2, "buy_qty": 100, "sell_qty": 100, "self_trades": 1}}


def write_events(folder) -> None:
    events = make_events(timestamp_ns=T0 + np.array([0, 4, 5]) * SEC, order_id=[1, 2, 3],
                         trader_id=[9, 42, 42], symbol_id=3, price=10.0, quantity=100,
                         event_type=2, side=[0, 0, 1])
    write_event_array(folder / "demo.events.bin", events)


def fake_engine(tmp_path, monkeypatch) -> list:
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        alerts = cmd[cmd.index("--alerts") + 1]
        with open(alerts, "w") as f:
            f.write(json.dumps(ALERT) + "\n")
        return subprocess.CompletedProcess(cmd, 0, stdout=SUMMARY, stderr="")

    monkeypatch.setattr(prepare.subprocess, "run", run)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)  # template notes, no local LLM
    return calls


def test_prepare_writes_engine_stats_and_review_rows(tmp_path, monkeypatch) -> None:
    write_events(tmp_path)
    calls = fake_engine(tmp_path, monkeypatch)

    assert prepare.main(["--data", str(tmp_path), "--name", "demo", "--engine", "tsa",
                         "--model", str(tmp_path / "no-model.onnx")]) == 0

    assert "--model" not in calls[0]  # a missing model file is simply not passed
    stats = json.loads((tmp_path / "demo.engine.json").read_text())
    assert stats["latency_p99_ns"] == 9000
    rows = [json.loads(line) for line in (tmp_path / "demo.review.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["note"]["checks_passed"]
    assert rows[0]["case"]["event_ids"] == [1, 2]


def test_http_note_maker_posts_the_case_to_the_note_service() -> None:
    service = NoteService(None)
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        case = Case.model_validate_json(request.content)
        return httpx.Response(200, text=service.make_note(case).model_dump_json())

    maker = prepare.HttpNoteMaker("http://notes:8000/", transport=httpx.MockTransport(handler))
    case = Case(alert_id=1, pattern="spoofing", trader_id=1, symbol_id=2, window_seconds=1,
                window_start_ns=0, window_end_ns=1, stats={"large_cancels": 3}, event_ids=[4])

    result = maker.make_note(case)

    assert seen == ["/v1/notes"]
    assert result.alert_id == 1 and result.checks_passed


def test_case_command_turns_alerts_into_cases(tmp_path) -> None:
    write_events(tmp_path)
    alerts = tmp_path / "demo.alerts.jsonl"
    alerts.write_text(json.dumps(ALERT) + "\n\n")
    out = tmp_path / "demo.cases.jsonl"

    assert case_main(["--data", str(tmp_path), "--name", "demo", "--alerts", str(alerts),
                      "--out", str(out)]) == 0

    cases = [Case.model_validate_json(line) for line in out.read_text().splitlines()]
    assert [c.alert_id for c in cases] == [1]
    assert cases[0].window_seconds == 60
