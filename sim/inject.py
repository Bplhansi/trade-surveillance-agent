"""Injectors: each adds one instance of a suspicious pattern to the market.

Every injector returns an Injection: the new events, plus a mask saying which of them
belong to the pattern. Innocent counterparties of the bad actor's fills are included
(so the market stays consistent) but are not labelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from sim.event_array import concat_events, make_events, take_events, time_order
from sim.event_format import EventType, Side
from sim.market import (
    TICK,
    OrderIds,
    PricePath,
    other_traders,
    round_to_lots,
    round_to_tick,
)

MS = 1_000_000  # nanoseconds in a millisecond
SECOND = 1_000_000_000


class Pattern(IntEnum):
    NONE = 0
    SPOOFING = 1
    WASH_TRADE = 2
    PUMP_AND_DUMP = 3

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class Injection:
    pattern: Pattern
    trader_id: int
    symbol_id: int
    events: np.ndarray
    is_pattern_event: np.ndarray  # bool per event: True = part of the misconduct

    @property
    def start_ns(self) -> int:
        return int(self.events["timestamp_ns"][self.is_pattern_event].min())

    @property
    def end_ns(self) -> int:
        return int(self.events["timestamp_ns"][self.is_pattern_event].max())


def _assemble(pattern: Pattern, trader_id: int, symbol_id: int,
              parts: list[tuple[np.ndarray, bool]]) -> Injection:
    events = concat_events([events for events, _ in parts])
    mask = np.concatenate([np.full(len(events), flag) for events, flag in parts])
    order = time_order(events)
    return Injection(pattern, trader_id, symbol_id, take_events(events, order), mask[order])


def _fills(rng: np.random.Generator, *, ts: np.ndarray, order_id: np.ndarray, trader_id: int,
           symbol_id: int, price: np.ndarray, qty: np.ndarray, side: Side,
           n_traders: int) -> list[tuple[np.ndarray, bool]]:
    """New orders by the bad actor that fill 1 ms later against innocent traders."""
    common = dict(order_id=order_id, symbol_id=symbol_id, price=price, quantity=qty)
    counterparties = other_traders(rng, trader_id, n_traders, size=len(ts))
    return [
        (make_events(timestamp_ns=ts, trader_id=trader_id, event_type=EventType.NEW,
                     side=side, **common), True),
        (make_events(timestamp_ns=ts + MS, trader_id=trader_id, event_type=EventType.TRADE,
                     side=side, **common), True),
        (make_events(timestamp_ns=ts + MS, trader_id=counterparties,
                     event_type=EventType.TRADE, side=1 - side, **common), False),
    ]


def inject_spoofing(rng: np.random.Generator, path: PricePath, ids: OrderIds, *,
                    trader_id: int, symbol_id: int, start_ns: int, n_bursts: int,
                    n_layers: int, layer_qty: int, n_traders: int) -> Injection:
    """Large orders placed near the price to mislead others, then cancelled.

    Each burst: n_layers large orders on one side, then a small real order on the other
    side that gets filled, then all the large orders are cancelled.
    """
    parts: list[tuple[np.ndarray, bool]] = []
    t = start_ns
    for _ in range(n_bursts):
        spoof_side = Side(int(rng.integers(0, 2)))
        real_side = Side(1 - spoof_side)
        mid = float(path.mid_at(symbol_id, t))
        direction = -1 if spoof_side == Side.BUY else 1  # buy orders sit below the mid

        layer_ts = t + np.arange(n_layers, dtype=np.int64) * 2 * MS
        layer_ids = ids.take(n_layers)
        layer_px = round_to_tick(mid + direction * (np.arange(n_layers) + 1) * TICK)
        layer_q = round_to_lots(layer_qty * rng.uniform(0.8, 1.2, n_layers))
        parts.append((make_events(timestamp_ns=layer_ts, order_id=layer_ids, trader_id=trader_id,
                                  symbol_id=symbol_id, price=layer_px, quantity=layer_q,
                                  event_type=EventType.NEW, side=spoof_side), True))

        real_ts = np.array([layer_ts[-1] + 20 * MS])
        parts += _fills(rng, ts=real_ts, order_id=ids.take(1), trader_id=trader_id,
                        symbol_id=symbol_id, price=round_to_tick([mid - direction * TICK]),
                        qty=round_to_lots(rng.uniform(50, 200, 1)), side=real_side,
                        n_traders=n_traders)

        cancel_ts = real_ts[0] + MS + rng.uniform(50, 400, n_layers).astype(np.int64) * MS
        parts.append((make_events(timestamp_ns=cancel_ts, order_id=layer_ids, trader_id=trader_id,
                                  symbol_id=symbol_id, price=layer_px, quantity=layer_q,
                                  event_type=EventType.CANCEL, side=spoof_side), True))
        t = int(cancel_ts.max()) + int(rng.uniform(1, 3) * SECOND)
    return _assemble(Pattern.SPOOFING, trader_id, symbol_id, parts)


def inject_wash_trade(rng: np.random.Generator, path: PricePath, ids: OrderIds, *,
                      trader_id: int, symbol_id: int, start_ns: int, n_rounds: int,
                      qty: int) -> Injection:
    """The same trader repeatedly buys from and sells to themselves at the same price."""
    parts: list[tuple[np.ndarray, bool]] = []
    t = start_ns
    for _ in range(n_rounds):
        price = float(round_to_tick(path.mid_at(symbol_id, t)))
        q = int(round_to_lots(qty * rng.uniform(0.8, 1.2)))
        buy_id, sell_id = ids.take(2)
        for ts, oid, etype, side in (
            (t, buy_id, EventType.NEW, Side.BUY),
            (t + MS, sell_id, EventType.NEW, Side.SELL),
            (t + 2 * MS, buy_id, EventType.TRADE, Side.BUY),
            (t + 2 * MS, sell_id, EventType.TRADE, Side.SELL),
        ):
            parts.append((make_events(timestamp_ns=ts, order_id=oid, trader_id=trader_id,
                                      symbol_id=symbol_id, price=price, quantity=q,
                                      event_type=etype, side=side), True))
        t += int(rng.uniform(0.5, 3) * SECOND)
    return _assemble(Pattern.WASH_TRADE, trader_id, symbol_id, parts)


def pump_bump(path: PricePath, *, symbol_id: int, start_ns: int, rise_ns: int, fall_ns: int,
              rise_pct: float) -> PricePath:
    """A copy of the price path where one symbol rises by rise_pct, then falls back."""
    grid_ts = path.start_ns + path.step_ns * np.arange(path.mids.shape[1], dtype=np.int64)
    elapsed = grid_ts - start_ns
    rising = np.clip(elapsed / rise_ns, 0, 1)
    falling = np.clip((elapsed - rise_ns) / fall_ns, 0, 1)
    factor = np.where((elapsed < 0) | (elapsed > rise_ns + fall_ns), 1.0,
                      1 + rise_pct * (rising - falling))
    mids = path.mids.copy()
    mids[symbol_id - 1] *= factor
    return PricePath(start_ns=path.start_ns, step_ns=path.step_ns, mids=mids)


def inject_pump_and_dump(rng: np.random.Generator, path: PricePath, ids: OrderIds, *,
                         trader_id: int, symbol_id: int, start_ns: int, rise_ns: int,
                         fall_ns: int, rise_pct: float, n_buys: int, n_sells: int,
                         buy_qty: int, n_traders: int) -> tuple[Injection, PricePath]:
    """One account buys heavily while the price is pushed up, then sells it all near the top.

    Returns the injection and the new price path (with the pump), which everyone else's
    orders should then be generated from.
    """
    bumped = pump_bump(path, symbol_id=symbol_id, start_ns=start_ns, rise_ns=rise_ns,
                       fall_ns=fall_ns, rise_pct=rise_pct)

    buy_ts = np.sort(start_ns + rng.uniform(0, rise_ns, n_buys).astype(np.int64))
    buy_q = round_to_lots(buy_qty * rng.uniform(0.8, 1.2, n_buys))
    buy_px = round_to_tick(bumped.mid_at(symbol_id, buy_ts) + TICK)

    # Sell the whole position in the first half of the fall, so it ends flat.
    sell_ts = np.sort(start_ns + rise_ns + rng.uniform(0, fall_ns / 2, n_sells).astype(np.int64))
    total = int(buy_q.sum())
    each = total // n_sells // 10 * 10
    sell_q = np.full(n_sells, each, dtype=np.uint32)
    sell_q[-1] = total - each * (n_sells - 1)
    sell_px = round_to_tick(bumped.mid_at(symbol_id, sell_ts) - TICK)

    parts = _fills(rng, ts=buy_ts, order_id=ids.take(n_buys), trader_id=trader_id,
                   symbol_id=symbol_id, price=buy_px, qty=buy_q, side=Side.BUY,
                   n_traders=n_traders)
    parts += _fills(rng, ts=sell_ts, order_id=ids.take(n_sells), trader_id=trader_id,
                    symbol_id=symbol_id, price=sell_px, qty=sell_q, side=Side.SELL,
                    n_traders=n_traders)
    return _assemble(Pattern.PUMP_AND_DUMP, trader_id, symbol_id, parts), bumped
