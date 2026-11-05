"""Build the train / validation / test datasets.

Usage (from the repo root):
    python -m sim.generate --out data                 # 1M train, 250k validation, 250k test
    python -m sim.generate --out data --train-events 100000

Each dataset is generated separately with its own seed (no slicing of one stream), and
the test set uses bad actors and pattern sizes that train/validation never contain, so a
model cannot score well by memorising them.

Files written per dataset (e.g. "train"):
    train.events.bin     events in the shared 40-byte format (readable by the C++ engine)
    train.labels.csv     one row per suspicious event (event_index is the row in events.bin)
    train.patterns.csv   one row per injected pattern instance
    train.manifest.json  seed, config, counts and SHA-256 of each file, for reproducibility
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import sim
from sim.event_array import concat_events, take_events, time_order, write_event_array
from sim.inject import (
    SECOND,
    Injection,
    Pattern,
    inject_pump_and_dump,
    inject_spoofing,
    inject_wash_trade,
)
from sim.market import (
    MarketConfig,
    OrderIds,
    generate_normal_events,
    make_price_path,
    market_duration_ns,
)

# How many instances of each pattern to inject per million events.
PATTERNS_PER_MILLION = {Pattern.SPOOFING: 60, Pattern.WASH_TRADE: 60, Pattern.PUMP_AND_DUMP: 15}


@dataclass(frozen=True)
class PatternSizes:
    """(low, high) ranges each pattern instance's size is drawn from."""

    spoof_bursts: tuple[int, int]
    spoof_layers: tuple[int, int]
    spoof_layer_qty: tuple[int, int]
    wash_rounds: tuple[int, int]
    wash_qty: tuple[int, int]
    pump_rise_s: tuple[int, int]
    pump_fall_s: tuple[int, int]
    pump_rise_pct: tuple[float, float]
    pump_buys: tuple[int, int]
    pump_sells: tuple[int, int]
    pump_buy_qty: tuple[int, int]


TRAIN_SIZES = PatternSizes(
    spoof_bursts=(2, 4), spoof_layers=(3, 5), spoof_layer_qty=(2000, 5000),
    wash_rounds=(6, 12), wash_qty=(200, 1000),
    pump_rise_s=(120, 300), pump_fall_s=(120, 300), pump_rise_pct=(0.04, 0.08),
    pump_buys=(30, 60), pump_sells=(10, 20), pump_buy_qty=(300, 800),
)

# Deliberately different: more, smaller spoof orders; fewer, smaller wash rounds;
# slower, gentler pumps with more, smaller buys.
TEST_SIZES = PatternSizes(
    spoof_bursts=(3, 6), spoof_layers=(6, 9), spoof_layer_qty=(800, 2000),
    wash_rounds=(4, 8), wash_qty=(100, 400),
    pump_rise_s=(300, 500), pump_fall_s=(200, 400), pump_rise_pct=(0.03, 0.05),
    pump_buys=(60, 100), pump_sells=(20, 40), pump_buy_qty=(150, 400),
)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    seed: int
    market: MarketConfig
    n_spoofing: int
    n_wash_trade: int
    n_pump_and_dump: int
    bad_trader_ids: tuple[int, int]  # inclusive range, inside the normal trader population
    sizes: PatternSizes


@dataclass(frozen=True)
class PatternRecord:
    pattern_id: int
    pattern: Pattern
    trader_id: int
    symbol_id: int
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class Dataset:
    config: DatasetConfig
    events: np.ndarray
    pattern_code: np.ndarray  # uint8 per event: Pattern value, 0 = normal
    pattern_id: np.ndarray  # int32 per event: PatternRecord id, -1 = normal
    patterns: list[PatternRecord]


def _count(n_events: int, pattern: Pattern) -> int:
    return max(1, round(n_events * PATTERNS_PER_MILLION[pattern] / 1_000_000))


def default_configs(train_events: int = 1_000_000) -> list[DatasetConfig]:
    def make(name: str, seed: int, n_events: int, bad_ids: tuple[int, int],
             sizes: PatternSizes) -> DatasetConfig:
        return DatasetConfig(
            name=name, seed=seed, market=MarketConfig(n_events=n_events),
            n_spoofing=_count(n_events, Pattern.SPOOFING),
            n_wash_trade=_count(n_events, Pattern.WASH_TRADE),
            n_pump_and_dump=_count(n_events, Pattern.PUMP_AND_DUMP),
            bad_trader_ids=bad_ids, sizes=sizes,
        )

    small = train_events // 4
    return [
        make("train", 42, train_events, (1, 250), TRAIN_SIZES),
        make("validation", 43, small, (1, 250), TRAIN_SIZES),
        make("test", 44, small, (251, 500), TEST_SIZES),
    ]


def _between(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(*bounds))


def _int_between(rng: np.random.Generator, bounds: tuple[int, int]) -> int:
    return int(rng.integers(bounds[0], bounds[1] + 1))


def _renumber_order_ids(events: np.ndarray) -> None:
    """Renumber order IDs 1, 2, 3... in order of first appearance (in place)."""
    unique_ids, first_index, inverse = np.unique(
        events["order_id"], return_index=True, return_inverse=True)
    new_ids = np.empty(len(unique_ids), dtype=np.uint32)
    new_ids[np.argsort(first_index)] = np.arange(1, len(unique_ids) + 1, dtype=np.uint32)
    events["order_id"] = new_ids[inverse]


def build_dataset(cfg: DatasetConfig) -> Dataset:
    rng = np.random.default_rng(cfg.seed)
    ids = OrderIds()
    market = cfg.market
    sizes = cfg.sizes
    path = make_price_path(rng, market)
    t0 = market.start_ns
    span = market_duration_ns(market)
    injections: list[Injection] = []

    def bad_actor() -> int:
        return _int_between(rng, cfg.bad_trader_ids)

    # 1. Pumps first: they change the price that everyone else then trades around.
    #    Each symbol's timeline is split into slots so pumps never overlap.
    slots_per_symbol = math.ceil(cfg.n_pump_and_dump / market.n_symbols)
    slot_len = int(span * 0.9) // slots_per_symbol
    for i in range(cfg.n_pump_and_dump):
        rise = _between(rng, sizes.pump_rise_s) * SECOND
        fall = _between(rng, sizes.pump_fall_s) * SECOND
        shrink = min(1.0, 0.8 * slot_len / (rise + fall))  # fit short (test-size) datasets
        rise, fall = int(rise * shrink), int(fall * shrink)
        slot_start = t0 + int(span * 0.05) + (i // market.n_symbols) * slot_len
        start = slot_start + int(rng.uniform(0, slot_len - rise - fall))
        injection, path = inject_pump_and_dump(
            rng, path, ids, trader_id=bad_actor(), symbol_id=i % market.n_symbols + 1,
            start_ns=start, rise_ns=rise, fall_ns=fall,
            rise_pct=_between(rng, sizes.pump_rise_pct),
            n_buys=_int_between(rng, sizes.pump_buys),
            n_sells=_int_between(rng, sizes.pump_sells),
            buy_qty=_int_between(rng, sizes.pump_buy_qty), n_traders=market.n_traders,
        )
        injections.append(injection)

    # 2. Normal market activity, using the pumped prices.
    normal = generate_normal_events(rng, market, path, ids)

    # 3. Short patterns at random times and symbols.
    def random_start() -> int:
        return t0 + int(rng.uniform(0.02, 0.9) * span)

    def random_symbol() -> int:
        return int(rng.integers(1, market.n_symbols + 1))

    for _ in range(cfg.n_spoofing):
        injections.append(inject_spoofing(
            rng, path, ids, trader_id=bad_actor(), symbol_id=random_symbol(),
            start_ns=random_start(), n_bursts=_int_between(rng, sizes.spoof_bursts),
            n_layers=_int_between(rng, sizes.spoof_layers),
            layer_qty=_int_between(rng, sizes.spoof_layer_qty), n_traders=market.n_traders,
        ))
    for _ in range(cfg.n_wash_trade):
        injections.append(inject_wash_trade(
            rng, path, ids, trader_id=bad_actor(), symbol_id=random_symbol(),
            start_ns=random_start(), n_rounds=_int_between(rng, sizes.wash_rounds),
            qty=_int_between(rng, sizes.wash_qty),
        ))

    # 4. Merge into one time-ordered stream, carrying the labels along.
    events = concat_events([normal] + [inj.events for inj in injections])
    pattern_code = np.concatenate(
        [np.zeros(len(normal), dtype=np.uint8)]
        + [np.where(inj.is_pattern_event, inj.pattern, 0).astype(np.uint8) for inj in injections])
    pattern_id = np.concatenate(
        [np.full(len(normal), -1, dtype=np.int32)]
        + [np.where(inj.is_pattern_event, pid, -1).astype(np.int32)
           for pid, inj in enumerate(injections)])
    order = time_order(events)
    events = take_events(events, order)
    pattern_code, pattern_id = pattern_code[order], pattern_id[order]
    _renumber_order_ids(events)

    patterns = [
        PatternRecord(pid, inj.pattern, inj.trader_id, inj.symbol_id, inj.start_ns, inj.end_ns)
        for pid, inj in enumerate(injections)
    ]
    return Dataset(cfg, events, pattern_code, pattern_id, patterns)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_dataset(ds: Dataset, out_dir: str | Path) -> Path:
    """Write the dataset's four files. Returns the manifest path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = ds.config.name
    events_path = out_dir / f"{name}.events.bin"
    labels_path = out_dir / f"{name}.labels.csv"
    patterns_path = out_dir / f"{name}.patterns.csv"
    manifest_path = out_dir / f"{name}.manifest.json"

    write_event_array(events_path, ds.events)

    labelled = np.flatnonzero(ds.pattern_code > 0)
    with labels_path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["event_index", "order_id", "trader_id", "pattern", "pattern_id"])
        for idx, oid, tid, code, pid in zip(
            labelled.tolist(), ds.events["order_id"][labelled].tolist(),
            ds.events["trader_id"][labelled].tolist(), ds.pattern_code[labelled].tolist(),
            ds.pattern_id[labelled].tolist(), strict=True,
        ):
            writer.writerow([idx, oid, tid, Pattern(code).label, pid])

    with patterns_path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["pattern_id", "pattern", "trader_id", "symbol_id", "start_ns", "end_ns"])
        for p in ds.patterns:
            writer.writerow([p.pattern_id, p.pattern.label, p.trader_id, p.symbol_id,
                             p.start_ns, p.end_ns])

    events_by_pattern = {Pattern(c).label: int((ds.pattern_code == c).sum()) for c in Pattern}
    manifest = {
        "name": name,
        "seed": ds.config.seed,
        "sim_version": sim.__version__,
        "event_format": "tsa-event-v1: 40-byte little-endian record (engine/include/tsa/event.hpp)",
        "n_events": len(ds.events),
        "events_by_pattern": events_by_pattern,
        "instances_by_pattern": {
            p.label: sum(r.pattern == p for r in ds.patterns) for p in Pattern if p != Pattern.NONE
        },
        "suspicious_fraction": round(float((ds.pattern_code > 0).mean()), 6),
        "config": asdict(ds.config),
        "sha256": {path.name: _sha256(path) for path in (events_path, labels_path, patterns_path)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate simulated trading datasets.")
    parser.add_argument("--out", default="data", help="output folder (default: data)")
    parser.add_argument("--train-events", type=int, default=1_000_000,
                        help="normal events in the train set; validation and test get 1/4")
    args = parser.parse_args(argv)

    for cfg in default_configs(args.train_events):
        started = time.perf_counter()
        ds = build_dataset(cfg)
        write_dataset(ds, args.out)
        seconds = time.perf_counter() - started
        suspicious = int((ds.pattern_code > 0).sum())
        print(f"{cfg.name:<10} {len(ds.events):>9,} events  {suspicious:>6,} suspicious "
              f"({suspicious / len(ds.events):.2%})  {len(ds.patterns):>3} patterns  "
              f"{seconds:.1f}s", file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
