"""Binary event format shared with the C++ engine (engine/include/tsa/event.hpp).

Each event is a fixed 40-byte little-endian record that mirrors the C++ struct:

    offset  size  field
         0     8  timestamp_ns  (uint64)
         8     4  order_id      (uint32)
        12     4  trader_id     (uint32)
        16     4  symbol_id     (uint32)
        20     4  (padding)
        24     8  price         (float64)
        32     4  quantity      (uint32)
        36     1  type          (uint8: 0=New, 1=Cancel, 2=Trade)
        37     1  side          (uint8: 0=Buy, 1=Sell)
        38     2  (padding)

If you change this layout, change the C++ struct too; tests on both sides will catch a mismatch.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

# "<" = little-endian with no automatic padding; "x" = one padding byte.
EVENT_STRUCT = struct.Struct("<QIII4xdIBB2x")
EVENT_SIZE = EVENT_STRUCT.size


class EventType(IntEnum):
    NEW = 0
    CANCEL = 1
    TRADE = 2


class Side(IntEnum):
    BUY = 0
    SELL = 1


@dataclass(frozen=True, slots=True)
class Event:
    timestamp_ns: int
    order_id: int
    trader_id: int
    symbol_id: int
    price: float
    quantity: int
    type: EventType
    side: Side

    def pack(self) -> bytes:
        return EVENT_STRUCT.pack(
            self.timestamp_ns,
            self.order_id,
            self.trader_id,
            self.symbol_id,
            self.price,
            self.quantity,
            self.type,
            self.side,
        )

    @classmethod
    def unpack(cls, record: bytes) -> Event:
        ts, order_id, trader_id, symbol_id, price, qty, etype, side = EVENT_STRUCT.unpack(record)
        return cls(ts, order_id, trader_id, symbol_id, price, qty, EventType(etype), Side(side))


def write_events(path: str | Path, events: Iterable[Event]) -> None:
    """Write events to a binary file, creating parent folders if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for event in events:
            f.write(event.pack())


def read_events(path: str | Path) -> list[Event]:
    """Read all events from a binary file."""
    data = Path(path).read_bytes()
    if len(data) % EVENT_SIZE != 0:
        raise ValueError(
            f"{path}: size {len(data)} bytes is not a multiple of {EVENT_SIZE} (one event)"
        )
    return [Event.unpack(record) for (record,) in struct.iter_unpack(f"{EVENT_SIZE}s", data)]


def sample_events() -> list[Event]:
    """Ten fixed events used to check that Python and C++ agree on the format.

    engine/tests/test_event.cpp builds the same ten events independently.
    """
    return [
        Event(
            timestamp_ns=1_700_000_000_000_000_000 + i * 1_000,
            order_id=1000 + i,
            trader_id=1 + i % 3,
            symbol_id=1,
            price=100.0 + 0.25 * i,
            quantity=100 * (i + 1),
            type=EventType(i % 3),
            side=Side(i % 2),
        )
        for i in range(10)
    ]
