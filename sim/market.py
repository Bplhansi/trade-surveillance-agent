"""Normal (innocent) market activity.

This is a simple statistical market, not a real exchange with a full order book:
- each symbol has a mid price that follows a random walk (one value per second);
- traders place orders a few ticks away from the mid price at random times;
- each order is later cancelled, filled, or left resting.
A fill produces two Trade events with the same order_id: one for the order's owner
and one for a different trader on the opposite side.

Innocent traders sometimes place very large orders and cancel them ("decoys"), so
large-order cancels alone are not proof of spoofing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sim.event_array import concat_events, make_events, take_events, time_order
from sim.event_format import EventType, Side

TICK = 0.01  # smallest price step: one cent
SECOND_NS = 1_000_000_000


@dataclass(frozen=True)
class MarketConfig:
    n_events: int = 1_000_000
    n_symbols: int = 5
    n_traders: int = 500
    start_ns: int = 1_767_605_400 * SECOND_NS  # 2026-01-05 09:30 UTC (simulated)
    mean_order_gap_ns: int = 11_000_000  # one new order every ~11 ms on average
    volatility_per_sqrt_s: float = 0.0005  # random-walk step size per second
    cancel_prob: float = 0.55
    fill_prob: float = 0.30
    mean_cancel_delay_ns: int = 300_000_000
    mean_fill_delay_ns: int = 800_000_000
    mean_qty: float = 100.0
    decoy_large_prob: float = 0.01  # share of innocent orders that are very large
    decoy_large_mult: int = 20
    decoy_large_cancel_prob: float = 0.8


def events_per_order(cfg: MarketConfig) -> float:
    """Average events per order: the New, plus a Cancel, or two Trades for a fill."""
    return 1 + cfg.cancel_prob + 2 * cfg.fill_prob


def market_duration_ns(cfg: MarketConfig) -> int:
    """Roughly how long the market must run to produce cfg.n_events normal events."""
    return int(cfg.n_events / events_per_order(cfg) * cfg.mean_order_gap_ns)


@dataclass(frozen=True)
class PricePath:
    """Mid price of every symbol, sampled once per step_ns. Row i is symbol_id i + 1."""

    start_ns: int
    step_ns: int
    mids: np.ndarray  # shape (n_symbols, n_steps)

    @property
    def end_ns(self) -> int:
        return self.start_ns + self.step_ns * (self.mids.shape[1] - 1)

    def index_at(self, ts: np.ndarray | int) -> np.ndarray:
        idx = (np.asarray(ts, dtype=np.int64) - self.start_ns) // self.step_ns
        return np.clip(idx, 0, self.mids.shape[1] - 1)

    def mid_at(self, symbol_id: np.ndarray | int, ts: np.ndarray | int) -> np.ndarray:
        return self.mids[np.asarray(symbol_id) - 1, self.index_at(ts)]


class OrderIds:
    """Hands out unique order IDs."""

    def __init__(self, start: int = 1) -> None:
        self._next = start

    def take(self, n: int) -> np.ndarray:
        ids = np.arange(self._next, self._next + n, dtype=np.uint32)
        self._next += n
        return ids


def round_to_tick(prices: np.ndarray | float) -> np.ndarray:
    return np.round(np.round(np.asarray(prices) / TICK) * TICK, 2)


def round_to_lots(qty: np.ndarray | float) -> np.ndarray:
    """Round quantities to multiples of 10 shares, minimum 10."""
    return np.maximum(10, np.round(np.asarray(qty) / 10) * 10).astype(np.uint32)


def other_traders(rng: np.random.Generator, trader_id: np.ndarray | int, n_traders: int,
                  size: int | None = None) -> np.ndarray:
    """Random traders in 1..n_traders, never equal to trader_id."""
    pick = rng.integers(1, n_traders, size=size)
    return np.where(pick >= trader_id, pick + 1, pick)


def make_price_path(rng: np.random.Generator, cfg: MarketConfig,
                    step_ns: int = SECOND_NS) -> PricePath:
    n_steps = int(market_duration_ns(cfg) * 1.2) // step_ns + 2  # 20% spare time
    start_prices = np.round(rng.uniform(20, 200, cfg.n_symbols), 2)
    shocks = rng.normal(0, cfg.volatility_per_sqrt_s * math.sqrt(step_ns / SECOND_NS),
                        (cfg.n_symbols, n_steps))
    shocks[:, 0] = 0.0
    mids = start_prices[:, None] * np.exp(np.cumsum(shocks, axis=1))
    return PricePath(start_ns=cfg.start_ns, step_ns=step_ns, mids=mids)


def generate_normal_events(rng: np.random.Generator, cfg: MarketConfig, path: PricePath,
                           ids: OrderIds) -> np.ndarray:
    """Exactly cfg.n_events innocent events, sorted by time."""
    n = math.ceil(cfg.n_events / events_per_order(cfg) * 1.15) + 50  # spare, trimmed below

    new_ts = cfg.start_ns + np.cumsum(rng.exponential(cfg.mean_order_gap_ns, n)).astype(np.int64)
    symbol = rng.integers(1, cfg.n_symbols + 1, n)
    trader = rng.integers(1, cfg.n_traders + 1, n)
    side = rng.integers(0, 2, n)
    order_id = ids.take(n)

    qty = rng.lognormal(math.log(cfg.mean_qty), 0.5, n)
    large = rng.random(n) < cfg.decoy_large_prob
    qty = round_to_lots(np.where(large, qty * cfg.decoy_large_mult, qty))

    # Buy orders sit below the mid price, sell orders above it.
    offset_ticks = rng.geometric(0.35, n)
    direction = np.where(side == Side.BUY, -1, 1)
    price = round_to_tick(path.mid_at(symbol, new_ts) + direction * offset_ticks * TICK)

    fate = rng.random(n)
    cancel = fate < cfg.cancel_prob
    fill = (fate >= cfg.cancel_prob) & (fate < cfg.cancel_prob + cfg.fill_prob)
    large_fate = rng.random(n)
    cancel = np.where(large, large_fate < cfg.decoy_large_cancel_prob, cancel)
    fill = np.where(large, (~cancel) & (large_fate < 0.9), fill)

    cancel_ts = new_ts + 1 + rng.exponential(cfg.mean_cancel_delay_ns, n).astype(np.int64)
    fill_ts = new_ts + 1 + rng.exponential(cfg.mean_fill_delay_ns, n).astype(np.int64)
    counterparty = other_traders(rng, trader, cfg.n_traders, size=n)

    def block(mask: np.ndarray, ts: np.ndarray, who: np.ndarray, etype: EventType,
              sides: np.ndarray) -> np.ndarray:
        return make_events(
            timestamp_ns=ts[mask], order_id=order_id[mask], trader_id=who[mask],
            symbol_id=symbol[mask], price=price[mask], quantity=qty[mask],
            event_type=etype, side=sides[mask],
        )

    everyone = np.ones(n, dtype=bool)
    events = concat_events([
        block(everyone, new_ts, trader, EventType.NEW, side),
        block(cancel, cancel_ts, trader, EventType.CANCEL, side),
        block(fill, fill_ts, trader, EventType.TRADE, side),
        block(fill, fill_ts, counterparty, EventType.TRADE, 1 - side),
    ])
    if len(events) < cfg.n_events:
        raise RuntimeError(f"generated only {len(events)} events, wanted {cfg.n_events}")
    return take_events(events, time_order(events)[: cfg.n_events])
