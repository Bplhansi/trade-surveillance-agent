"""First Python test. Proves pytest can find and import the simulator package."""

import sim


def test_sim_reports_current_version() -> None:
    assert sim.__version__ == "0.1.0"
