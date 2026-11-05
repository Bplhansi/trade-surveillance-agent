"""Draw each symbol's trade price over time with the injected patterns marked.

Usage (from the repo root, after sim.generate):
    python -m sim.plot --data data --name train --out docs/sim.png
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file; no window needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from sim.event_array import read_event_array  # noqa: E402
from sim.event_format import EventType  # noqa: E402

STYLE = {
    "spoofing": {"color": "#d62728", "marker": "v", "name": "Spoofing"},
    "wash_trade": {"color": "#1f77b4", "marker": "o", "name": "Wash trade"},
    "pump_and_dump": {"color": "#ff7f0e", "marker": None, "name": "Pump and dump"},
}
NS_PER_MINUTE = 60e9


def _read_patterns(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def plot_dataset(data_dir: str | Path, name: str, out_path: str | Path,
                 max_points: int = 4000) -> None:
    data_dir = Path(data_dir)
    events = read_event_array(data_dir / f"{name}.events.bin")
    patterns = _read_patterns(data_dir / f"{name}.patterns.csv")

    t0 = int(events["timestamp_ns"].min())
    trades = events[events["type"] == EventType.TRADE]
    symbols = sorted(int(s) for s in np.unique(trades["symbol_id"]))

    fig, axes = plt.subplots(len(symbols), 1, figsize=(13, 2.3 * len(symbols)),
                             sharex=True, squeeze=False)
    for ax, symbol in zip(axes[:, 0], symbols, strict=True):
        mine = trades[trades["symbol_id"] == symbol]
        minutes = (mine["timestamp_ns"].astype(np.int64) - t0) / NS_PER_MINUTE
        prices = mine["price"]
        step = max(1, len(mine) // max_points)  # thin out points so the chart stays readable
        ax.plot(minutes[::step], prices[::step], color="0.35", lw=0.6)

        for p in (p for p in patterns if int(p["symbol_id"]) == symbol):
            style = STYLE[p["pattern"]]
            start = (int(p["start_ns"]) - t0) / NS_PER_MINUTE
            end = (int(p["end_ns"]) - t0) / NS_PER_MINUTE
            if style["marker"] is None:
                ax.axvspan(start, end, color=style["color"], alpha=0.25, lw=0)
            else:
                ax.scatter([start], [np.interp(start, minutes, prices)], s=36,
                           marker=style["marker"], color=style["color"],
                           edgecolors="black", linewidths=0.4, zorder=3)
        ax.set_ylabel(f"Symbol {symbol}\nprice")
        ax.grid(alpha=0.3)

    counts = {key: sum(p["pattern"] == key for p in patterns) for key in STYLE}
    handles = [
        Line2D([], [], color="0.35", lw=1, label="Trade price"),
        Line2D([], [], ls="", marker="v", color=STYLE["spoofing"]["color"], markeredgecolor="black",
               label=f"Spoofing ({counts['spoofing']})"),
        Line2D([], [], ls="", marker="o", color=STYLE["wash_trade"]["color"],
               markeredgecolor="black", label=f"Wash trade ({counts['wash_trade']})"),
        Patch(color=STYLE["pump_and_dump"]["color"], alpha=0.25,
              label=f"Pump and dump ({counts['pump_and_dump']})"),
    ]
    axes[0, 0].legend(handles=handles, loc="upper left", ncol=4, fontsize=8, framealpha=0.9)
    axes[0, 0].set_title(f"Simulated market: {name} set, {len(events):,} events, "
                         f"{len(patterns)} injected patterns")
    axes[-1, 0].set_xlabel("Minutes since market open")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot a simulated dataset.")
    parser.add_argument("--data", default="data", help="folder with the generated files")
    parser.add_argument("--name", default="train", help="dataset name (train/validation/test)")
    parser.add_argument("--out", default="docs/sim.png", help="output PNG file")
    args = parser.parse_args(argv)
    plot_dataset(args.data, args.name, args.out)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
