"""Tests for normal (non-suspicious) market activity."""

import numpy as np

from sim.event_format import EventType, Side
from sim.market import TICK, MarketConfig, OrderIds, generate_normal_events, make_price_path

CFG = MarketConfig(n_events=20_000, n_traders=50)


def _normal(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    path = make_price_path(rng, CFG)
    return generate_normal_events(rng, CFG, path, OrderIds())


def test_price_path_has_one_positive_row_per_symbol() -> None:
    path = make_price_path(np.random.default_rng(0), CFG)

    assert path.mids.shape[0] == CFG.n_symbols
    assert (path.mids > 0).all()


def test_mid_at_looks_up_the_price_for_that_second() -> None:
    path = make_price_path(np.random.default_rng(0), CFG)
    ts = path.start_ns + 3 * path.step_ns + 5

    assert path.mid_at(2, ts) == path.mids[1, 3]


def test_generates_exactly_the_requested_number_of_events() -> None:
    assert len(_normal()) == CFG.n_events


def test_same_seed_gives_identical_bytes() -> None:
    assert _normal(seed=1).tobytes() == _normal(seed=1).tobytes()


def test_different_seed_gives_different_events() -> None:
    assert _normal(seed=1).tobytes() != _normal(seed=2).tobytes()


def test_timestamps_never_go_backwards() -> None:
    ts = _normal()["timestamp_ns"].astype(np.int64)

    assert (np.diff(ts) >= 0).all()


def test_new_order_ids_are_unique() -> None:
    events = _normal()
    new_ids = events["order_id"][events["type"] == EventType.NEW]

    assert len(np.unique(new_ids)) == len(new_ids)


def test_every_cancel_follows_its_own_new_order() -> None:
    events = _normal()
    news = events[events["type"] == EventType.NEW]
    placed = {int(o): (int(t), int(tr)) for o, t, tr in news[["order_id", "timestamp_ns", "trader_id"]]}

    cancels = events[events["type"] == EventType.CANCEL]
    assert len(cancels) > 0
    for order_id, ts, trader in cancels[["order_id", "timestamp_ns", "trader_id"]]:
        new_ts, new_trader = placed[int(order_id)]
        assert ts > new_ts
        assert trader == new_trader


def test_every_fill_has_the_owner_and_a_different_counterparty() -> None:
    events = _normal()
    owners = {int(o): int(tr) for o, tr in events[events["type"] == EventType.NEW][["order_id", "trader_id"]]}
    trades = events[events["type"] == EventType.TRADE]
    # Trades whose New fell outside the kept window can't be checked; skip them.
    trades = trades[np.isin(trades["order_id"], list(owners))]

    order_ids, counts = np.unique(trades["order_id"], return_counts=True)
    complete = order_ids[counts == 2]
    assert len(complete) > 0
    for order_id in complete[:500]:
        pair = trades[trades["order_id"] == order_id]
        assert pair["side"][0] != pair["side"][1]
        assert pair["trader_id"][0] != pair["trader_id"][1]
        assert owners[int(order_id)] in pair["trader_id"]
        assert pair["price"][0] == pair["price"][1]
        assert pair["quantity"][0] == pair["quantity"][1]


def test_prices_are_positive_and_on_the_one_cent_grid() -> None:
    prices = _normal()["price"]

    assert (prices > 0).all()
    np.testing.assert_allclose(np.round(prices / TICK) * TICK, prices, atol=1e-9)


def test_ids_stay_in_range() -> None:
    events = _normal()

    assert events["symbol_id"].min() >= 1 and events["symbol_id"].max() <= CFG.n_symbols
    assert events["trader_id"].min() >= 1 and events["trader_id"].max() <= CFG.n_traders
    assert set(np.unique(events["side"])) <= {Side.BUY, Side.SELL}


def test_includes_large_decoy_orders_that_innocent_traders_cancel() -> None:
    events = _normal()
    news = events[events["type"] == EventType.NEW]
    large_ids = news["order_id"][news["quantity"] >= 10 * np.median(news["quantity"])]
    cancelled = events["order_id"][events["type"] == EventType.CANCEL]

    assert len(large_ids) > 0
    assert np.isin(large_ids, cancelled).any()
