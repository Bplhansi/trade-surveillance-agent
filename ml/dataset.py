"""Load the feature rows exported by the C++ engine and the per-event labels."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

# Class 0 = normal; the rest match the pattern names in labels.csv / patterns.csv.
CLASSES = ("none", "spoofing", "wash_trade", "pump_and_dump")


def load_features(data: Path, name: str) -> tuple[np.ndarray, list[str]]:
    """Return (rows x columns float32 array, column names) from <name>.features.bin/.json."""
    meta = json.loads((data / f"{name}.features.json").read_text())
    if meta["dtype"] != "float32":
        raise ValueError(f"unsupported dtype {meta['dtype']!r}")
    columns = list(meta["columns"])
    flat = np.fromfile(data / f"{name}.features.bin", dtype="<f4")
    if flat.size != meta["rows"] * len(columns):
        raise ValueError(
            f"{name}.features.bin has {flat.size} values, expected {meta['rows']} rows "
            f"x {len(columns)} columns")
    return flat.reshape(meta["rows"], len(columns)), columns


def load_labels(data: Path, name: str, n_events: int) -> np.ndarray:
    """Class index (see CLASSES) for every event; 0 for events that are not part of a pattern."""
    y = np.zeros(n_events, dtype=np.int64)
    with (data / f"{name}.labels.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            y[int(row["event_index"])] = CLASSES.index(row["pattern"])
    return y
