"""Tests for building the train / validation / test datasets."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sim.event_array import EVENT_DTYPE
from sim.event_format import read_events
from sim.generate import Dataset, build_dataset, default_configs, main, write_dataset
from sim.inject import Pattern

SMALL_TRAIN_EVENTS = 40_000


@pytest.fixture(scope="module")
def configs():
    return default_configs(train_events=SMALL_TRAIN_EVENTS)


@pytest.fixture(scope="module")
def train(configs) -> Dataset:
    return build_dataset(configs[0])


def test_default_configs_are_train_validation_test(configs) -> None:
    assert [c.name for c in configs] == ["train", "validation", "test"]
    assert configs[0].market.n_events == SMALL_TRAIN_EVENTS
    assert len({c.seed for c in configs}) == 3


def test_test_set_uses_new_bad_actors_and_different_pattern_sizes(configs) -> None:
    train_cfg, val_cfg, test_cfg = configs
    train_lo, train_hi = train_cfg.bad_trader_ids
    test_lo, test_hi = test_cfg.bad_trader_ids

    assert train_hi < test_lo or test_hi < train_lo  # no overlap
    assert val_cfg.bad_trader_ids == train_cfg.bad_trader_ids
    assert test_cfg.sizes != train_cfg.sizes


def test_bad_actors_are_drawn_from_the_normal_trader_population(configs) -> None:
    for cfg in configs:
        lo, hi = cfg.bad_trader_ids
        assert 1 <= lo <= hi <= cfg.market.n_traders


def test_every_configured_pattern_is_injected(train: Dataset) -> None:
    cfg = train.config
    found = [p.pattern for p in train.patterns]

    assert found.count(Pattern.SPOOFING) == cfg.n_spoofing
    assert found.count(Pattern.WASH_TRADE) == cfg.n_wash_trade
    assert found.count(Pattern.PUMP_AND_DUMP) == cfg.n_pump_and_dump


def test_labels_point_at_events_by_the_patterns_trader(train: Dataset) -> None:
    by_id = {p.pattern_id: p for p in train.patterns}
    labelled = np.flatnonzero(train.pattern_id >= 0)

    assert len(labelled) > 0
    for idx in labelled:
        record = by_id[int(train.pattern_id[idx])]
        assert train.events["trader_id"][idx] == record.trader_id
        assert train.events["symbol_id"][idx] == record.symbol_id
        assert train.pattern_code[idx] == record.pattern


def test_bad_actor_trader_ids_come_from_the_configured_range(train: Dataset) -> None:
    lo, hi = train.config.bad_trader_ids

    assert all(lo <= p.trader_id <= hi for p in train.patterns)


def test_suspicious_events_are_rare(train: Dataset) -> None:
    fraction = (train.pattern_code > 0).mean()

    assert 0 < fraction < 0.05


def test_events_keep_the_shared_40_byte_layout(train: Dataset) -> None:
    assert train.events.dtype == EVENT_DTYPE


def test_events_are_in_time_order(train: Dataset) -> None:
    ts = train.events["timestamp_ns"].astype(np.int64)

    assert (np.diff(ts) >= 0).all()


def test_order_ids_are_numbered_in_order_of_first_appearance(train: Dataset) -> None:
    # Stops a model "cheating" by spotting injected orders from their ID range.
    _, first_index = np.unique(train.events["order_id"], return_index=True)

    assert (np.diff(first_index) > 0).all()


def test_write_dataset_creates_the_four_files(tmp_path: Path, train: Dataset) -> None:
    write_dataset(train, tmp_path)

    for suffix in ("events.bin", "labels.csv", "patterns.csv", "manifest.json"):
        assert (tmp_path / f"train.{suffix}").exists()


def test_written_events_are_readable_in_the_shared_format(tmp_path: Path, train: Dataset) -> None:
    write_dataset(train, tmp_path)

    assert len(read_events(tmp_path / "train.events.bin")) == len(train.events)


def test_labels_csv_lists_every_suspicious_event(tmp_path: Path, train: Dataset) -> None:
    write_dataset(train, tmp_path)
    with (tmp_path / "train.labels.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == int((train.pattern_code > 0).sum())
    assert set(rows[0]) == {"event_index", "order_id", "trader_id", "pattern", "pattern_id"}
    assert {r["pattern"] for r in rows} == {"spoofing", "wash_trade", "pump_and_dump"}


def test_manifest_records_seed_counts_and_file_hashes(tmp_path: Path, train: Dataset) -> None:
    write_dataset(train, tmp_path)
    manifest = json.loads((tmp_path / "train.manifest.json").read_text())

    assert manifest["seed"] == train.config.seed
    assert manifest["n_events"] == len(train.events)
    for filename, digest in manifest["sha256"].items():
        assert hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest() == digest


def test_same_config_gives_byte_identical_files(tmp_path: Path, configs) -> None:
    for out in ("a", "b"):
        write_dataset(build_dataset(configs[1]), tmp_path / out)

    for f in (tmp_path / "a").iterdir():
        assert f.read_bytes() == (tmp_path / "b" / f.name).read_bytes(), f.name


def test_command_line_writes_all_three_datasets(tmp_path: Path) -> None:
    exit_code = main(["--out", str(tmp_path), "--train-events", "20000"])

    assert exit_code == 0
    for name in ("train", "validation", "test"):
        assert (tmp_path / f"{name}.events.bin").exists()
