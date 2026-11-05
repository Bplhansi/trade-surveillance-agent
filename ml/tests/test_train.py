"""Smoke test for ml/train.py on a tiny dataset with an obvious signal."""

import json
from pathlib import Path

import numpy as np
import pytest

from ml.dataset import CLASSES
from ml.train import main
from sim.event_array import make_events, write_event_array

SEC = 1_000_000_000
COLUMNS = ["signal", "noise"]


def write_split(data: Path, name: str, seed: int, n: int = 600) -> None:
    rng = np.random.default_rng(seed)
    ts = np.arange(n, dtype=np.uint64) * SEC + 1_000 * SEC
    trader = rng.integers(1, 20, n).astype(np.uint32)
    trader[100:110] = 99  # the bad actor: a spoofing case
    X = np.column_stack([np.zeros(n), rng.normal(size=n)]).astype(np.float32)
    X[100:110, 0] = 5.0
    X.tofile(data / f"{name}.features.bin")
    (data / f"{name}.features.json").write_text(
        json.dumps({"rows": n, "dtype": "float32", "columns": COLUMNS}))
    write_event_array(data / f"{name}.events.bin",
                      make_events(timestamp_ns=ts, order_id=np.arange(1, n + 1), trader_id=trader,
                                  symbol_id=1, price=10.0, quantity=100, event_type=1, side=0))
    (data / f"{name}.labels.csv").write_text(
        "event_index,order_id,trader_id,pattern,pattern_id\n"
        + "".join(f"{i},{i + 1},99,spoofing,0\n" for i in range(100, 110)))
    (data / f"{name}.patterns.csv").write_text(
        "pattern_id,pattern,trader_id,symbol_id,start_ns,end_ns\n"
        f"0,spoofing,99,1,{int(ts[100])},{int(ts[109])}\n")


def test_train_picks_thresholds_on_validation_and_tests_only_when_asked(tmp_path: Path, capsys) -> None:
    for i, name in enumerate(("train", "validation", "test")):
        write_split(tmp_path, name, seed=i)
    out = tmp_path / "models"

    assert main(["--data", str(tmp_path), "--out", str(out)]) == 0
    first = capsys.readouterr().out
    assert "validation" in first and "dataset: test" not in first

    meta = json.loads((out / "model_card.json").read_text())
    assert meta["columns"] == COLUMNS
    assert meta["classes"] == list(CLASSES)
    assert set(meta["models"]) == {"isolation_forest", "gradient_boosting"}
    assert meta["models"]["gradient_boosting"]["validation"]["overall"]["recall"] == 1.0
    assert (out / "gradient_boosting.joblib").exists()

    assert main(["--data", str(tmp_path), "--out", str(out), "--test"]) == 0
    assert "dataset: test" in capsys.readouterr().out
    meta = json.loads((out / "model_card.json").read_text())
    assert "test" in meta["models"]["gradient_boosting"]
    assert meta["models"]["gradient_boosting"]["n_iter"] >= 1
    assert "numpy" in meta["versions"] and "train.features.bin" in meta["data_sha256"]

    # The test set is scored once: neither a second --test nor re-training may overwrite it.
    with pytest.raises(SystemExit, match="already"):
        main(["--data", str(tmp_path), "--out", str(out), "--test"])
    with pytest.raises(SystemExit, match="already"):
        main(["--data", str(tmp_path), "--out", str(out)])
