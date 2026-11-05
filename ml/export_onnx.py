"""Export the gradient boosted model to ONNX for the C++ engine.

The ONNX model takes float32 feature rows [N, 33] (same columns as the engine exports) and
returns class probabilities [N, 4]. The engine computes the same score as Python:
score = -log10(P(normal)), pattern = most likely non-normal class.

Usage:
    python -m ml.export_onnx --model models/gradient_boosting.joblib --out models/gradient_boosting.onnx
    python -m ml.export_onnx --parity-dir <dir>   # tiny model + expected scores, for C++ tests
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import HistGradientBoostingClassifier

from ml.dataset import CLASSES
from ml.train import scores_and_rules

N_TINY_FEATURES = 33  # same width as the engine's FeatureRow, so C++ tests can reuse it


def export_onnx(model: HistGradientBoostingClassifier, n_features: int, path: Path) -> None:
    onx = convert_sklearn(
        model,
        initial_types=[("features", FloatTensorType([None, n_features]))],
        options={id(model): {"zipmap": False}},  # plain probability tensor, not a list of maps
        target_opset={"": 17, "ai.onnx.ml": 3},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onx.SerializeToString())


def onnx_scores(session, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Score rows with an onnxruntime session, the way the C++ engine does."""
    proba = session.run(["probabilities"], {"features": X.astype(np.float32)})[0]
    score = -np.log10(np.maximum(proba[:, 0].astype(np.float64), 1e-30))
    pattern = 1 + np.argmax(proba[:, 1:], axis=1)
    return score, np.asarray(CLASSES, dtype=object)[pattern]


def train_tiny_model(seed: int = 0) -> tuple[HistGradientBoostingClassifier, np.ndarray]:
    """A small 4-class model on random data: enough to test export and scoring parity."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(800, N_TINY_FEATURES)).astype(np.float32)
    y = np.zeros(len(X), dtype=np.int64)
    y[X[:, 0] > 1.0] = 1
    y[X[:, 1] > 1.2] = 2
    y[X[:, 2] > 1.4] = 3
    model = HistGradientBoostingClassifier(max_iter=30, random_state=seed, early_stopping=False)
    return model.fit(X, y), X


def write_parity_files(out: Path) -> None:
    """parity.onnx + rows (float32) + expected scores (float64) + patterns (int32) from Python."""
    out.mkdir(parents=True, exist_ok=True)
    model, X = train_tiny_model(seed=0)
    export_onnx(model, X.shape[1], out / "parity.onnx")
    rows = X[:50]
    scores, rules = scores_and_rules("gradient_boosting", model, rows)
    rows.astype("<f4").tofile(out / "parity.rows.bin")
    scores.astype("<f8").tofile(out / "parity.scores.bin")
    np.array([CLASSES.index(r) for r in rules], dtype="<i4").tofile(out / "parity.patterns.bin")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the trained model to ONNX.")
    parser.add_argument("--model", type=Path, default=Path("models/gradient_boosting.joblib"))
    parser.add_argument("--out", type=Path, default=Path("models/gradient_boosting.onnx"))
    parser.add_argument("--parity-dir", type=Path, help="write the tiny parity model instead")
    args = parser.parse_args(argv)
    if args.parity_dir:
        write_parity_files(args.parity_dir)
        print(f"parity files written to: {args.parity_dir}")
        return 0
    model = joblib.load(args.model)
    export_onnx(model, model.n_features_in_, args.out)
    print(f"ONNX model written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
