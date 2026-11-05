"""Turn an engine alert (one line of <name>.alerts.jsonl) into a Case for the note service.

The case window is the alert's rolling window (1 s or 60 s) ending at the alert. event_ids
are the rows of this trader + symbol inside that window, newest 50 kept.

Usage:
    python -m service.case --data data --name test --alerts data/test.alerts.jsonl --out data/test.cases.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from service.schemas import Case
from sim.event_array import read_event_array

MAX_EVENT_IDS = 50
SEC = 1_000_000_000


def build_case(alert: dict, events: np.ndarray) -> Case:
    seconds = 1 if alert["window"] == "1s" else 60
    end = int(alert["timestamp_ns"])
    start = end - seconds * SEC
    ts = events["timestamp_ns"]
    lo, hi = np.searchsorted(ts, start, "left"), np.searchsorted(ts, end, "right")
    window = events[lo:hi]
    mine = np.flatnonzero((window["trader_id"] == alert["trader_id"])
                          & (window["symbol_id"] == alert["symbol_id"])) + lo
    ids = [int(i) for i in mine]
    return Case(
        alert_id=int(alert["alert_id"]),
        pattern=alert["rule"],
        trader_id=int(alert["trader_id"]),
        symbol_id=int(alert["symbol_id"]),
        source=alert.get("source", "rules"),
        model_score=alert.get("model_score"),
        window_seconds=seconds,
        window_start_ns=int(ts[mine[0]]) if len(mine) else start,
        window_end_ns=end,
        stats={k: float(v) for k, v in alert["evidence"].items()},
        event_ids=ids[-MAX_EVENT_IDS:],
        event_ids_truncated=len(ids) > MAX_EVENT_IDS,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build note-service cases from engine alerts.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--alerts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    events = read_event_array(args.data / f"{args.name}.events.bin")
    with args.alerts.open() as f, args.out.open("w") as out:
        n = 0
        for line in f:
            if line.strip():
                out.write(build_case(json.loads(line), events).model_dump_json() + "\n")
                n += 1
    print(f"{n} cases written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
