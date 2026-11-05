"""Train the anomaly models on engine features.

- Isolation Forest: unsupervised, never sees labels. Flags unusual rows as "anomaly".
- Gradient boosting (HistGradientBoostingClassifier): learns the 4 classes from the labels.

Each model's decision threshold is chosen on the VALIDATION set (best case-level F1). The
TEST set is only scored with --test, using the saved thresholds, and must be run once.

Usage:
    python -m ml.train --data data --out models           # train + validation
    python -m ml.train --data data --out models --test    # ... and the one-time test score
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

from ml.dataset import CLASSES, load_features, load_labels
from ml.evaluate_alerts import ANOMALY, PatternScore, format_table, load_cases, score
from ml.model_alerts import EventKeys, alerts_from_scores, choose_threshold, quantile_candidates

SEED = 0
MODELS = ("isolation_forest", "gradient_boosting")


def fit(name: str, X: np.ndarray, y: np.ndarray):
    if name == "isolation_forest":
        return IsolationForest(n_estimators=200, max_samples=256, random_state=SEED).fit(X)
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_leaf_nodes=31, class_weight="balanced",
        early_stopping=False,  # no hidden internal hold-out: every train row is used
        random_state=SEED).fit(X, y)


def scores_and_rules(name: str, model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-event score (higher = more suspicious) and the alert's rule name."""
    if name == "isolation_forest":
        return -model.score_samples(X), np.full(len(X), ANOMALY)
    classes = np.asarray(model.classes_)
    if not set(classes.tolist()) <= set(range(len(CLASSES))):
        raise ValueError(f"unexpected classes {classes}; expected a subset of 0..{len(CLASSES) - 1}")
    proba = np.zeros((len(X), len(CLASSES)))
    proba[:, classes] = model.predict_proba(X)  # place each column by its class label
    pattern = 1 + np.argmax(proba[:, 1:], axis=1)
    # -log10 P(normal): same ranking as 1 - P(normal), but without the subtraction that
    # makes scores near 1 indistinguishable in float32 (the C++/ONNX side).
    score = -np.log10(np.maximum(proba[:, 0], 1e-30))
    return score, np.asarray(CLASSES, dtype=object)[pattern]


class Split:
    def __init__(self, data: Path, name: str) -> None:
        self.name = name
        self.X, self.columns = load_features(data, name)
        self.n = len(self.X)
        self.keys = EventKeys.from_events_file(data / f"{name}.events.bin")
        self.cases = load_cases(data / f"{name}.patterns.csv")
        if len(self.keys.timestamp_ns) != self.n:
            raise ValueError(f"{name}: {self.n} feature rows but {len(self.keys.timestamp_ns)} events")


def as_dict(scores: dict[str, PatternScore]) -> dict:
    return {k: {"precision": s.precision, "recall": s.recall, "caught": s.caught,
                "cases": s.cases, "alerts": s.alerts, "false_alarms": s.false_positives,
                "false_alarms_per_10k": s.false_alarms_per_10k} for k, s in scores.items()}


def report(split: Split, model_name: str, model, threshold: float) -> dict:
    s, rules = scores_and_rules(model_name, model, split.X)
    result = score(alerts_from_scores(s, rules, split.keys, threshold), split.cases, split.n)
    print(f"\n[{model_name}] dataset: {split.name}  |  {split.n:,} events  |  threshold {threshold:.6g}")
    for line in format_table(result):
        print(line)
    return as_dict(result)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate the anomaly models.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--test", action="store_true", help="also score the test set (run once)")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    card_path = args.out / "model_card.json"

    if args.test:  # use the saved models and thresholds; nothing is re-fitted or re-tuned
        card = json.loads(card_path.read_text())
        if any("test" in m for m in card["models"].values()):
            raise SystemExit(f"{card_path}: the test set was already scored; it is scored once only")
        test = Split(args.data, "test")
        for name in MODELS:
            model = joblib.load(args.out / f"{name}.joblib")
            card["models"][name]["test"] = report(test, name, model,
                                                  card["models"][name]["threshold"])
        card_path.write_text(json.dumps(card, indent=2))
        return 0

    if card_path.exists() and any("test" in m for m in
                                   json.loads(card_path.read_text())["models"].values()):
        raise SystemExit(f"{card_path}: test results already exist; re-training would hide them. "
                         "Move the folder away first if you really mean to start over.")
    train, val = Split(args.data, "train"), Split(args.data, "validation")
    y_train = load_labels(args.data, "train", train.n)
    card = {"columns": train.columns, "classes": list(CLASSES), "seed": SEED,
            "sklearn": sklearn.__version__, "trained_on": "train",
            "threshold_chosen_on": "validation (best case-level F1, middle of ties)",
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "sklearn": sklearn.__version__, "joblib": joblib.__version__},
            "data_sha256": {f"{n}.{kind}": sha256(args.data / f"{n}.{kind}")
                            for n in ("train", "validation")
                            for kind in ("features.bin", "events.bin", "labels.csv")},
            "models": {}}
    for name in MODELS:
        model = fit(name, train.X, y_train)
        joblib.dump(model, args.out / f"{name}.joblib")
        s, rules = scores_and_rules(name, model, val.X)
        threshold, f1 = choose_threshold(s, rules, val.keys, val.cases, val.n,
                                         quantile_candidates(s))
        card["models"][name] = {"threshold": threshold, "validation_f1": f1,
                                "n_iter": int(getattr(model, "n_iter_", 0)) or None,
                                "validation": report(val, name, model, threshold)}
    card_path.write_text(json.dumps(card, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
