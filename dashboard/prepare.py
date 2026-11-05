"""Get everything the dashboard shows ready: run the engine, build cases, write notes.

Writes data/<name>.alerts.jsonl (by the engine), data/<name>.engine.json (speed summary)
and data/<name>.review.jsonl (one row per alert: alert, case, note).

Usage (PowerShell):
    .\\.venv\\Scripts\\python.exe -m dashboard.prepare --name validation
    ... --service-url http://127.0.0.1:8000   # get notes from the running note service
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from service.case import build_case
from service.schemas import Case, NoteResult
from sim.event_array import read_event_array

_SUMMARY_KEYS = {
    "events": "events", "throughput": "throughput_events_per_s",
    "latency mean": "latency_mean_ns", "latency p50": "latency_p50_ns",
    "latency p95": "latency_p95_ns", "latency p99": "latency_p99_ns",
    "latency max": "latency_max_ns", "alerts": "alerts",
}
_LINE = re.compile(r"^([a-z][a-z0-9 ]*?)\s+([\d,]+)\b")


class NoteMaker(Protocol):
    def make_note(self, case: Case) -> NoteResult: ...


class HttpNoteMaker:
    """Gets notes from the running note service (used by Docker Compose)."""

    def __init__(self, url: str, timeout_s: float = 600.0, transport=None) -> None:
        import httpx  # only needed in this mode

        self.client = httpx.Client(base_url=url.rstrip("/"), timeout=timeout_s,
                                   transport=transport)  # transport: a stand-in for tests

    def make_note(self, case: Case) -> NoteResult:
        response = self.client.post("/v1/notes", content=case.model_dump_json(),
                                    headers={"content-type": "application/json"})
        response.raise_for_status()
        return NoteResult.model_validate_json(response.text)


def parse_engine_summary(text: str) -> dict[str, int]:
    """Numbers from the 'summary' block that tsa_replay prints at the end."""
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "summary")
    except StopIteration:
        raise ValueError("no 'summary' block in the engine output") from None
    stats = {}
    for line in lines[start + 1:]:
        m = _LINE.match(line.strip())
        if m and m.group(1) in _SUMMARY_KEYS:
            stats[_SUMMARY_KEYS[m.group(1)]] = int(m.group(2).replace(",", ""))
    return stats


def build_review_rows(alerts: list[dict], events: np.ndarray, notes: NoteMaker) -> list[dict]:
    rows = []
    for a in alerts:
        case = build_case(a, events)
        rows.append({"alert_id": a["alert_id"], "alert": a, "case": case.model_dump(),
                     "note": notes.make_note(case).model_dump()})
    return rows


def _default_engine() -> Path:
    exe = "tsa_replay.exe" if sys.platform == "win32" else "tsa_replay"
    sub = Path("Release") if sys.platform == "win32" else Path()
    return Path("engine") / "build" / sub / exe


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare alerts, notes and stats for review.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--name", default="validation")
    parser.add_argument("--engine", type=Path, default=_default_engine())
    parser.add_argument("--rules", type=Path, default=Path("config/rules.ini"))
    parser.add_argument("--model", type=Path, default=Path("models/gradient_boosting.onnx"))
    parser.add_argument("--service-url", help="note service URL (default: make notes in-process)")
    args = parser.parse_args(argv)

    events_path = args.data / f"{args.name}.events.bin"
    alerts_path = args.data / f"{args.name}.alerts.jsonl"
    cmd = [str(args.engine), str(events_path), "--rules", str(args.rules),
           "--alerts", str(alerts_path)]
    if args.model and args.model.exists():
        cmd += ["--model", str(args.model)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    stats = parse_engine_summary(out)
    (args.data / f"{args.name}.engine.json").write_text(json.dumps(stats, indent=2))

    if args.service_url:
        notes: NoteMaker = HttpNoteMaker(args.service_url)
    else:
        from dotenv import load_dotenv

        from service.llm import generator_from_env
        from service.notes import NoteService

        load_dotenv()
        notes = NoteService(generator_from_env())
    alerts = [json.loads(line) for line in alerts_path.read_text().splitlines() if line.strip()]
    rows = build_review_rows(alerts, read_event_array(events_path), notes)
    review_path = args.data / f"{args.name}.review.jsonl"
    review_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    by_source = {s: sum(r["note"]["source"] == s for r in rows) for s in ("llm", "template")}
    passed = sum(r["note"]["checks_passed"] for r in rows)
    print(f"engine: {stats.get('throughput_events_per_s', 0):,} events/s, "
          f"p99 {stats.get('latency_p99_ns', 0):,} ns")
    print(f"{len(rows)} alerts -> {review_path}  notes: {by_source}  passing checks: {passed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
