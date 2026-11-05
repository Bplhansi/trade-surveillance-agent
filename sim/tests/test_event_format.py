"""Tests for the binary event format shared by the Python simulator and the C++ engine."""

import struct
from pathlib import Path

import pytest

from sim.event_format import (
    EVENT_SIZE,
    Event,
    EventType,
    Side,
    read_events,
    sample_events,
    write_events,
)
from sim.write_sample_events import main as write_sample_events_main


def test_each_event_record_is_40_bytes() -> None:
    assert EVENT_SIZE == 40


def test_enum_numbers_match_the_cpp_engine() -> None:
    assert [EventType.NEW, EventType.CANCEL, EventType.TRADE] == [0, 1, 2]
    assert [Side.BUY, Side.SELL] == [0, 1]


def test_sample_events_are_ten_known_events() -> None:
    events = sample_events()

    assert len(events) == 10
    assert events[0] == Event(
        timestamp_ns=1_700_000_000_000_000_000,
        order_id=1000,
        trader_id=1,
        symbol_id=1,
        price=100.0,
        quantity=100,
        type=EventType.NEW,
        side=Side.BUY,
    )
    assert events[9] == Event(
        timestamp_ns=1_700_000_000_000_009_000,
        order_id=1009,
        trader_id=1,
        symbol_id=1,
        price=102.25,
        quantity=1000,
        type=EventType.NEW,
        side=Side.SELL,
    )


def test_write_then_read_returns_the_same_events(tmp_path: Path) -> None:
    path = tmp_path / "events.bin"
    events = sample_events()

    write_events(path, events)

    assert read_events(path) == events


def test_fields_sit_at_the_byte_offsets_cpp_expects(tmp_path: Path) -> None:
    # Byte offsets of each field inside one 40-byte C++ Event struct.
    path = tmp_path / "events.bin"
    event = sample_events()[5]

    write_events(path, [event])
    raw = path.read_bytes()

    assert len(raw) == 40
    assert struct.unpack_from("<Q", raw, 0)[0] == event.timestamp_ns
    assert struct.unpack_from("<I", raw, 8)[0] == event.order_id
    assert struct.unpack_from("<I", raw, 12)[0] == event.trader_id
    assert struct.unpack_from("<I", raw, 16)[0] == event.symbol_id
    assert struct.unpack_from("<d", raw, 24)[0] == event.price
    assert struct.unpack_from("<I", raw, 32)[0] == event.quantity
    assert raw[36] == event.type
    assert raw[37] == event.side


def test_read_rejects_a_file_with_a_partial_event(tmp_path: Path) -> None:
    path = tmp_path / "broken.bin"
    path.write_bytes(b"\x00" * 41)

    with pytest.raises(ValueError, match="multiple of 40"):
        read_events(path)


def test_script_writes_the_ten_sample_events(tmp_path: Path) -> None:
    path = tmp_path / "out" / "sample.bin"

    exit_code = write_sample_events_main([str(path)])

    assert exit_code == 0
    assert read_events(path) == sample_events()
