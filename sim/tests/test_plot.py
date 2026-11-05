"""Tests for the price chart with injected patterns marked."""

from pathlib import Path

from sim.generate import build_dataset, default_configs, write_dataset
from sim.plot import plot_dataset


def test_plot_saves_a_png(tmp_path: Path) -> None:
    write_dataset(build_dataset(default_configs(train_events=20_000)[0]), tmp_path)
    out = tmp_path / "chart.png"

    plot_dataset(tmp_path, "train", out)

    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
