"""Write the ten sample events to a file, for the C++ round-trip test.

Usage (from the repo root):
    python -m sim.write_sample_events <output-file>
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from sim.event_format import sample_events, write_events


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m sim.write_sample_events <output-file>", file=sys.stderr)
        return 2

    events = sample_events()
    write_events(args[0], events)
    print(f"wrote {len(events)} events to {args[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
