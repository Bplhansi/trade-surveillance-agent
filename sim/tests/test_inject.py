"""Tests for the three suspicious-pattern injectors."""

import numpy as np
import pytest

from sim.event_format import EventType, Side
from sim.inject import Pattern, inject_pump_and_dump, inject_spoofing, inject_wash_trade, pump_bump
from sim.market import MarketConfig, OrderIds, PricePath, make_price_path

SECOND = 1_000_000_000
CFG = MarketConfig(n_events=200_000)


@pytest.fixture
def path() -> PricePath:
    return make_price_path(np.random.default_rng(0), CFG)


def _spoof(path: PricePath, seed: int = 1):
    return inject_spoofing(
        np.random.default_rng(seed), path, OrderIds(),
        trader_id=7, symbol_id=2, start_ns=path.start_ns + 10 * SECOND,
        n_bursts=2, n_layers=4, layer_qty=3000, n_traders=50,
    )


def _pump(path: PricePath, seed: int = 1):
    return inject_pump_and_dump(
        np.random.default_rng(seed), path, OrderIds(),
        trader_id=9, symbol_id=1, start_ns=path.start_ns + 60 * SECOND,
        rise_ns=120 * SECOND, fall_ns=120 * SECOND, rise_pct=0.06,
        n_buys=20, n_sells=10, buy_qty=800, n_traders=50,
    )


# --- Spoofing -----------------------------------------------------------------

def test_spoofing_large_orders_are_all_cancelled(path: PricePath) -> None:
    inj = _spoof(path)
    mine = inj.events[inj.is_pattern_event]
    large = mine[(mine["type"] == EventType.NEW) & (mine["quantity"] >= 1000)]
    cancels = mine[mine["type"] == EventType.CANCEL]

    assert inj.pattern == Pattern.SPOOFING
    assert len(large) == 2 * 4  # bursts x layers
    assert set(cancels["order_id"].tolist()) == set(large["order_id"].tolist())
    for order_id, new_ts in large[["order_id", "timestamp_ns"]]:
        assert cancels["timestamp_ns"][cancels["order_id"] == order_id][0] > new_ts


def test_spoofing_small_real_order_fills_on_the_other_side(path: PricePath) -> None:
    inj = _spoof(path)
    mine = inj.events[inj.is_pattern_event]
    large_news = mine[(mine["type"] == EventType.NEW) & (mine["quantity"] >= 1000)]
    my_trades = mine[mine["type"] == EventType.TRADE]

    assert len(my_trades) == 2  # one small fill per burst
    assert (my_trades["quantity"] < 1000).all()
    for trade in my_trades:
        # The large orders placed just before this fill sit on the opposite side.
        before = large_news[large_news["timestamp_ns"] < trade["timestamp_ns"]][-4:]
        assert (before["side"] != trade["side"]).all()


def test_spoofing_only_the_spoofer_is_labelled(path: PricePath) -> None:
    inj = _spoof(path)
    mine = inj.events[inj.is_pattern_event]
    others = inj.events[~inj.is_pattern_event]

    assert (mine["trader_id"] == 7).all() and (mine["symbol_id"] == 2).all()
    assert len(others) == 2  # innocent counterparties of the two fills
    assert (others["type"] == EventType.TRADE).all() and (others["trader_id"] != 7).all()


# --- Wash trading -------------------------------------------------------------

def test_wash_trade_buys_and_sells_to_itself_with_no_net_position(path: PricePath) -> None:
    inj = inject_wash_trade(
        np.random.default_rng(1), path, OrderIds(),
        trader_id=5, symbol_id=3, start_ns=path.start_ns + 10 * SECOND, n_rounds=5, qty=500,
    )
    trades = inj.events[inj.events["type"] == EventType.TRADE]

    assert inj.pattern == Pattern.WASH_TRADE
    assert inj.is_pattern_event.all()
    assert (inj.events["trader_id"] == 5).all()
    assert len(trades) == 10
    buys = trades[trades["side"] == Side.BUY]
    sells = trades[trades["side"] == Side.SELL]
    assert buys["quantity"].sum() == sells["quantity"].sum()
    np.testing.assert_array_equal(buys["timestamp_ns"], sells["timestamp_ns"])
    np.testing.assert_array_equal(buys["price"], sells["price"])
    np.testing.assert_array_equal(buys["quantity"], sells["quantity"])


# --- Pump and dump -------------------------------------------------------------

def test_pump_bump_raises_the_price_then_returns_it(path: PricePath) -> None:
    start = path.start_ns + 60 * SECOND
    bumped = pump_bump(path, symbol_id=1, start_ns=start, rise_ns=60 * SECOND,
                       fall_ns=60 * SECOND, rise_pct=0.05)
    i0 = 60  # grid index of start (1-second grid)

    assert bumped.mids[0, i0 + 60] == pytest.approx(path.mids[0, i0 + 60] * 1.05)
    assert bumped.mids[0, i0 - 1] == path.mids[0, i0 - 1]
    assert bumped.mids[0, i0 + 121] == path.mids[0, i0 + 121]
    np.testing.assert_array_equal(bumped.mids[1:], path.mids[1:])  # other symbols untouched


def test_pump_bump_does_not_change_the_original_path(path: PricePath) -> None:
    before = path.mids.copy()

    pump_bump(path, symbol_id=1, start_ns=path.start_ns, rise_ns=SECOND * 10,
              fall_ns=SECOND * 10, rise_pct=0.05)

    np.testing.assert_array_equal(path.mids, before)


def test_pump_and_dump_buys_into_the_rise_then_sells_everything(path: PricePath) -> None:
    inj, bumped = _pump(path)
    mine = inj.events[inj.is_pattern_event]
    trades = mine[mine["type"] == EventType.TRADE]
    buys = trades[trades["side"] == Side.BUY]
    sells = trades[trades["side"] == Side.SELL]

    assert inj.pattern == Pattern.PUMP_AND_DUMP
    assert (mine["trader_id"] == 9).all()
    assert len(buys) == 20 and len(sells) == 10
    assert buys["timestamp_ns"].max() < sells["timestamp_ns"].min()
    assert buys["quantity"].sum() == sells["quantity"].sum()
    assert buys["price"][-1] > buys["price"][0]  # pushed the price up
    assert bumped.mids[0].max() > path.mids[0].max()


# --- Shared behaviour ---------------------------------------------------------

def test_injection_window_matches_its_labelled_events(path: PricePath) -> None:
    inj = _spoof(path)
    mine = inj.events[inj.is_pattern_event]

    assert inj.start_ns == mine["timestamp_ns"].min()
    assert inj.end_ns == mine["timestamp_ns"].max()


def test_injectors_are_repeatable_with_the_same_seed(path: PricePath) -> None:
    assert _spoof(path, seed=3).events.tobytes() == _spoof(path, seed=3).events.tobytes()
    assert _pump(path, seed=3)[0].events.tobytes() == _pump(path, seed=3)[0].events.tobytes()


def test_injected_events_are_in_time_order(path: PricePath) -> None:
    for inj in (_spoof(path), _pump(path)[0]):
        ts = inj.events["timestamp_ns"].astype(np.int64)
        assert (np.diff(ts) >= 0).all()
