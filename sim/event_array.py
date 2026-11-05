"""Fast NumPy version of the shared event format.

EVENT_DTYPE describes the same 40-byte record as sim/event_format.py and the C++ struct,
so a whole array of events can be written or read in one go instead of one at a time.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from sim.event_format import EVENT_SIZE

EVENT_DTYPE = np.dtype(
    {
        "names": ["timestamp_ns", "order_id", "trader_id", "symbol_id",
                  "price", "quantity", "type", "side"],
        "formats": ["<u8", "<u4", "<u4", "<u4", "<f8", "<u4", "u1", "u1"],
        "offsets": [0, 8, 12, 16, 24, 32, 36, 37],
        "itemsize": EVENT_SIZE,
    }
)


def empty_events(n: int) -> np.ndarray:
    """An array of n zeroed events (padding bytes are zero too)."""
    return np.zeros(n, dtype=EVENT_DTYPE)


def make_events(
    *,
    timestamp_ns: ArrayLike,
    order_id: ArrayLike,
    trader_id: ArrayLike,
    symbol_id: ArrayLike,
    price: ArrayLike,
    quantity: ArrayLike,
    event_type: ArrayLike,
    side: ArrayLike,
) -> np.ndarray:
    """Build an event array from columns. Single values are repeated for every row."""
    columns = {
        "timestamp_ns": timestamp_ns, "order_id": order_id, "trader_id": trader_id,
        "symbol_id": symbol_id, "price": price, "quantity": quantity,
        "type": event_type, "side": side,
    }
    n = max(np.size(value) for value in columns.values())
    events = empty_events(n)
    for name, value in columns.items():
        events[name] = value
    return events


def as_event_array(events: np.ndarray) -> np.ndarray:
    """A copy of events in the exact 40-byte layout, with all padding bytes set to zero.

    NumPy operations such as np.concatenate can return a "packed" version of the dtype
    with the padding removed. Converting back with astype() leaves the padding bytes
    uninitialised (random), so copy field by field into a zeroed array instead.
    """
    out = empty_events(len(events))
    for name in EVENT_DTYPE.names:
        out[name] = events[name]
    return out


def concat_events(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Join event arrays, keeping the 40-byte layout (unlike plain np.concatenate)."""
    return as_event_array(np.concatenate(arrays))


def time_order(events: np.ndarray) -> np.ndarray:
    """Row order that sorts events by time; ties keep their original order."""
    return np.argsort(events["timestamp_ns"], kind="stable")


def take_events(events: np.ndarray, index: np.ndarray) -> np.ndarray:
    """events[index] in the 40-byte layout with zeroed padding (plain indexing may not be)."""
    return as_event_array(events[index])


def write_event_array(path: str | Path, events: np.ndarray) -> None:
    """Write events to a binary file, creating parent folders if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    as_event_array(events).tofile(path)


def read_event_array(path: str | Path) -> np.ndarray:
    """Read every event in a binary file into an array."""
    size = Path(path).stat().st_size
    if size % EVENT_SIZE != 0:
        raise ValueError(
            f"{path}: size {size} bytes is not a multiple of {EVENT_SIZE} (one event)"
        )
    return np.fromfile(path, dtype=EVENT_DTYPE)
