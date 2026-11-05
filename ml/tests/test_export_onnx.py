"""Tests for exporting the gradient boosted model to ONNX."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from ml.export_onnx import N_TINY_FEATURES, export_onnx, onnx_scores, train_tiny_model, write_parity_files
from ml.train import scores_and_rules


def test_onnx_scores_match_scikit_learn(tmp_path: Path) -> None:
    model, X = train_tiny_model(seed=1)
    path = tmp_path / "tiny.onnx"

    export_onnx(model, X.shape[1], path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got_scores, got_patterns = onnx_scores(session, X)
    want_scores, want_rules = scores_and_rules("gradient_boosting", model, X)

    np.testing.assert_allclose(got_scores, want_scores, atol=1e-3)
    assert (got_patterns == want_rules).all()


def test_tiny_model_knows_all_four_classes() -> None:
    model, X = train_tiny_model(seed=1)

    assert list(model.classes_) == [0, 1, 2, 3]
    assert X.shape[1] == N_TINY_FEATURES


def test_parity_files_hold_rows_and_expected_scores(tmp_path: Path) -> None:
    write_parity_files(tmp_path)

    rows = np.fromfile(tmp_path / "parity.rows.bin", dtype="<f4").reshape(-1, N_TINY_FEATURES)
    scores = np.fromfile(tmp_path / "parity.scores.bin", dtype="<f8")
    patterns = np.fromfile(tmp_path / "parity.patterns.bin", dtype="<i4")
    assert (tmp_path / "parity.onnx").exists()
    assert len(rows) == len(scores) == len(patterns) > 0
    assert set(patterns.tolist()) <= {1, 2, 3}
