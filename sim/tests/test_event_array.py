"""Tests for the fast NumPy version of the event format."""

from pathlib import Path

import numpy as np
import pytest

from sim.event_array import (
    EVENT_DTYPE,
    concat_events,
    make_events,
    read_event_array,
    take_events,
    write_event_array,
)
from sim.event_format import EVENT_SIZE, EventType, Side, read_events, sample_events


def _sample_array() -> np.ndarray:
    events = sample_events()
    return make_events(
        timestamp_ns=[e.timestamp_ns for e in events],
        order_id=[e.order_id for e in events],
        trader_id=[e.trader_id for e in events],
        symbol_id=[e.symbol_id for e in events],
        price=[e.price for e in events],
        quantity=[e.quantity for e in events],
        event_type=[e.type for e in events],
        side=[e.side for e in events],
    )


def test_numpy_record_is_40_bytes() -> None:
    assert EVENT_DTYPE.itemsize == EVENT_SIZE == 40


def test_numpy_bytes_match_the_struct_format_exactly() -> None:
    expected = b"".join(event.pack() for event in sample_events())

    assert _sample_array().tobytes() == expected


def test_write_then_read_array_returns_the_same_events(tmp_path: Path) -> None:
    path = tmp_path / "events.bin"
    events = _sample_array()

    write_event_array(path, events)

    np.testing.assert_array_equal(read_event_array(path), events)


def test_array_file_is_readable_by_the_struct_reader(tmp_path: Path) -> None:
    path = tmp_path / "events.bin"

    write_event_array(path, _sample_array())

    assert read_events(path) == sample_events()


def test_make_events_repeats_single_values_for_every_row() -> None:
    events = make_events(
        timestamp_ns=[1, 2, 3],
        order_id=7,
        trader_id=1,
        symbol_id=1,
        price=10.0,
        quantity=5,
        event_type=EventType.NEW,
        side=Side.BUY,
    )

    assert len(events) == 3
    assert (events["order_id"] == 7).all()


def test_concat_events_keeps_the_40_byte_layout() -> None:
    # Plain np.concatenate silently drops the padding (34-byte records).
    joined = concat_events([_sample_array(), _sample_array()])

    assert joined.dtype == EVENT_DTYPE
    assert len(joined.tobytes()) == 2 * 10 * 40


def test_written_padding_bytes_are_always_zero(tmp_path: Path) -> None:
    # Regression: a packed (padding-free) array used to be written with random padding.
    path = tmp_path / "events.bin"
    packed = np.concatenate([_sample_array(), _sample_array()])

    write_event_array(path, packed)

    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8).reshape(-1, 40)
    assert (raw[:, 20:24] == 0).all() and (raw[:, 38:40] == 0).all()
    np.testing.assert_array_equal(read_event_array(path), concat_events([_sample_array()] * 2))


def test_take_events_reorders_with_zeroed_padding() -> None:
    events = _sample_array()
    reordered = take_events(events, np.arange(len(events))[::-1])

    raw = np.frombuffer(reordered.tobytes(), dtype=np.uint8).reshape(-1, 40)
    assert (raw[:, 20:24] == 0).all() and (raw[:, 38:40] == 0).all()
    assert reordered["order_id"].tolist() == events["order_id"].tolist()[::-1]


def test_read_array_rejects_a_partial_event(tmp_path: Path) -> None:
    path = tmp_path / "broken.bin"
    path.write_bytes(b"\x00" * 41)

    with pytest.raises(ValueError, match="multiple of 40"):
        read_event_array(path)
