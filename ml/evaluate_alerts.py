"""Grade the engine's alerts against the injected cases.

Grading is per *case*, not per event: a case is one injected pattern instance (one row of
<name>.patterns.csv). An alert matches a case when it has the same pattern, trader and symbol,
and its time is inside the case (widened by GRACE_NS on each side).

- recall    = cases with at least one matching alert / all cases
- precision = alerts that match some case / all alerts   (None if there are no alerts)

Usage:
    python -m ml.evaluate_alerts --data data --name validation --alerts data/validation.alerts.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

PATTERNS = ("spoofing", "wash_trade", "pump_and_dump")
# A pattern-agnostic alert (e.g. from the Isolation Forest): matches a case of any pattern.
ANOMALY = "anomaly"
GRACE_NS = 10 * 1_000_000_000  # a rule may need a few seconds of evidence after the case ends
EVENT_SIZE = 40  # bytes per event record (docs/data.md)


@dataclass(frozen=True)
class Case:
    pattern_id: int
    pattern: str
    trader_id: int
    symbol_id: int
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class Alert:
    rule: str
    trader_id: int
    symbol_id: int
    timestamp_ns: int


@dataclass(frozen=True)
class PatternScore:
    alerts: int
    true_positives: int  # alerts that match a case
    cases: int
    caught: int  # cases with at least one matching alert
    n_events: int

    @property
    def false_positives(self) -> int:
        return self.alerts - self.true_positives

    @property
    def precision(self) -> float | None:
        return self.true_positives / self.alerts if self.alerts else None

    @property
    def recall(self) -> float | None:
        return self.caught / self.cases if self.cases else None

    @property
    def false_alarms_per_10k(self) -> float:
        return self.false_positives * 10_000 / self.n_events if self.n_events else 0.0


def load_cases(path: Path) -> list[Case]:
    with path.open(newline="") as f:
        return [
            Case(
                pattern_id=int(row["pattern_id"]),
                pattern=row["pattern"],
                trader_id=int(row["trader_id"]),
                symbol_id=int(row["symbol_id"]),
                start_ns=int(row["start_ns"]),
                end_ns=int(row["end_ns"]),
            )
            for row in csv.DictReader(f)
        ]


def load_alerts(path: Path, source: str | None = None) -> list[Alert]:
    """Alerts from a .jsonl file; with `source` ("rules"/"model") only those from that source.

    Files written before the model was added have no "source" field: those alerts all came from the rules.
    """
    alerts = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if source is not None and row.get("source", "rules") != source:
                continue
            if row["rule"] not in (*PATTERNS, ANOMALY):
                raise ValueError(f"{path}:{line_no}: unknown rule {row['rule']!r}")
            alerts.append(
                Alert(
                    rule=row["rule"],
                    trader_id=int(row["trader_id"]),
                    symbol_id=int(row["symbol_id"]),
                    timestamp_ns=int(row["timestamp_ns"]),
                )
            )
    return alerts


def _matches(alert: Alert, case: Case, grace_ns: int) -> bool:
    return (
        alert.rule in (case.pattern, ANOMALY)
        and alert.trader_id == case.trader_id
        and alert.symbol_id == case.symbol_id
        and case.start_ns - grace_ns <= alert.timestamp_ns <= case.end_ns + grace_ns
    )


def score(
    alerts: Sequence[Alert], cases: Sequence[Case], n_events: int, grace_ns: int = GRACE_NS
) -> dict[str, PatternScore]:
    """Precision/recall per pattern, plus an "overall" row adding all patterns together."""
    by_key: dict[tuple[str, int, int], list[Case]] = {}
    for case in cases:
        by_key.setdefault((case.pattern, case.trader_id, case.symbol_id), []).append(case)

    hits: set[int] = set()  # pattern_ids caught
    true_positives = {p: 0 for p in PATTERNS}
    alert_counts = {p: 0 for p in (*PATTERNS, ANOMALY)}
    true_positives[ANOMALY] = 0
    for alert in alerts:
        alert_counts[alert.rule] += 1
        patterns = PATTERNS if alert.rule == ANOMALY else (alert.rule,)
        matched = [
            c for p in patterns for c in by_key.get((p, alert.trader_id, alert.symbol_id), [])
            if _matches(alert, c, grace_ns)
        ]
        if matched:
            true_positives[alert.rule] += 1
            hits.update(c.pattern_id for c in matched)

    result = {
        p: PatternScore(
            alerts=alert_counts[p],
            true_positives=true_positives[p],
            cases=sum(c.pattern == p for c in cases),
            caught=sum(c.pattern == p and c.pattern_id in hits for c in cases),
            n_events=n_events,
        )
        for p in PATTERNS
    }
    result["overall"] = PatternScore(
        alerts=sum(s.alerts for s in result.values()) + alert_counts[ANOMALY],
        true_positives=sum(s.true_positives for s in result.values()) + true_positives[ANOMALY],
        cases=sum(s.cases for s in result.values()),
        caught=sum(s.caught for s in result.values()),
        n_events=n_events,
    )
    return result


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def format_table(scores: dict[str, PatternScore]) -> Iterable[str]:
    yield (f"{'pattern':<15}{'precision':>10}{'recall':>8}{'caught':>10}"
           f"{'alerts':>8}{'false alarms':>14}{'per 10k events':>16}")
    for name, s in scores.items():
        yield (f"{name:<15}{_fmt(s.precision):>10}{_fmt(s.recall):>8}"
               f"{f'{s.caught}/{s.cases}':>10}{s.alerts:>8}{s.false_positives:>14}"
               f"{s.false_alarms_per_10k:>16.2f}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data"), help="folder with the dataset")
    parser.add_argument("--name", required=True, help="dataset name, e.g. validation")
    parser.add_argument("--alerts", type=Path, required=True, help="alerts .jsonl from tsa_replay")
    parser.add_argument("--source", choices=("rules", "model"),
                        help="grade only alerts from this source (default: all)")
    args = parser.parse_args(argv)

    n_events = (args.data / f"{args.name}.events.bin").stat().st_size // EVENT_SIZE
    cases = load_cases(args.data / f"{args.name}.patterns.csv")
    scores = score(load_alerts(args.alerts, args.source), cases, n_events)

    print(f"dataset: {args.name}  |  {n_events:,} events  |  {len(cases)} injected cases  |  "
          f"grace {GRACE_NS // 1_000_000_000} s  |  alerts from: {args.source or 'all sources'}")
    for line in format_table(scores):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
