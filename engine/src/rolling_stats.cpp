#include "tsa/rolling_stats.hpp"

#include <algorithm>

namespace tsa {

namespace {

std::uint64_t pair_key(std::uint32_t trader_id, std::uint32_t symbol_id) noexcept {
    return (static_cast<std::uint64_t>(trader_id) << 32) | symbol_id;
}

// Two Trade events are the two sides of one fill if they happen at the same moment,
// on the same symbol, at the same price and size, on opposite sides.
bool same_fill(const Event& a, const Event& b) noexcept {
    return a.timestamp_ns == b.timestamp_ns && a.symbol_id == b.symbol_id &&
           a.price == b.price && a.quantity == b.quantity && a.side != b.side;
}

}  // namespace

void RollingStats::apply(Bucket& b, const Event& e, bool self_trade) const {
    const bool large = e.quantity >= large_order_qty_;
    switch (e.type) {
        case EventType::New:
            ++b.orders;
            b.order_qty += e.quantity;
            b.max_order_qty = std::max(b.max_order_qty, e.quantity);
            if (large) {
                ++b.large_orders;
            }
            break;
        case EventType::Cancel:
            ++b.cancels;
            if (large) {
                ++b.large_cancels;
            }
            break;
        case EventType::Trade:
            if (b.trades == 0) {
                b.first_price = e.price;
            }
            b.last_price = e.price;
            ++b.trades;
            (e.side == Side::Buy ? b.buy_qty : b.sell_qty) += e.quantity;
            if (self_trade) {
                ++b.self_trades;
            }
            break;
    }
}

void RollingStats::update(const Event& e) {
    if (e.timestamp_ns != last_ts_ns_) {
        unmatched_trades_.clear();  // halves of one fill always share a timestamp
    }
    last_ts_ns_ = e.timestamp_ns;

    bool self_trade = false;
    if (e.type == EventType::Trade) {
        const auto partner = std::find_if(unmatched_trades_.begin(), unmatched_trades_.end(),
                                          [&](const Event& half) { return same_fill(half, e); });
        if (partner != unmatched_trades_.end()) {
            self_trade = partner->trader_id == e.trader_id;
            unmatched_trades_.erase(partner);
        } else {
            unmatched_trades_.push_back(e);
        }
    }

    PairWindows& w = pairs_[pair_key(e.trader_id, e.symbol_id)];
    apply(w.short_window.slot(static_cast<std::int64_t>(e.timestamp_ns / kShortBucketNs)), e,
          self_trade);
    apply(w.long_window.slot(static_cast<std::int64_t>(e.timestamp_ns / kLongBucketNs)), e,
          self_trade);
}

template <std::size_t N>
WindowStats RollingStats::summarize(const RingBuffer<Bucket, N>& ring, std::uint64_t bucket_ns,
                                    std::uint64_t now_ns) {
    WindowStats s;
    std::uint64_t order_qty = 0;
    bool have_trade = false;
    double first_price = 0.0;
    double last_price = 0.0;

    const auto now_index = static_cast<std::int64_t>(now_ns / bucket_ns);
    for (auto index = now_index - static_cast<std::int64_t>(N) + 1; index <= now_index; ++index) {
        const Bucket* b = ring.find(index);
        if (b == nullptr) {
            continue;
        }
        s.orders += b->orders;
        s.cancels += b->cancels;
        s.trades += b->trades;
        s.self_trades += b->self_trades;
        s.max_order_qty = std::max(s.max_order_qty, b->max_order_qty);
        s.large_orders += b->large_orders;
        s.large_cancels += b->large_cancels;
        s.buy_qty += b->buy_qty;
        s.sell_qty += b->sell_qty;
        order_qty += b->order_qty;
        if (b->trades > 0) {
            if (!have_trade) {
                first_price = b->first_price;
                have_trade = true;
            }
            last_price = b->last_price;
        }
    }

    if (s.orders > 0) {
        s.cancel_ratio = static_cast<double>(s.cancels) / s.orders;
        s.avg_order_qty = static_cast<double>(order_qty) / s.orders;
    }
    if (s.large_orders > 0) {
        s.large_cancel_ratio = static_cast<double>(s.large_cancels) / s.large_orders;
    }
    s.net_qty = static_cast<std::int64_t>(s.buy_qty) - static_cast<std::int64_t>(s.sell_qty);
    s.first_trade_price = have_trade ? first_price : 0.0;
    s.price_change = have_trade ? last_price - first_price : 0.0;
    return s;
}

WindowStats RollingStats::stats(std::uint32_t trader_id, std::uint32_t symbol_id, Window window,
                                std::uint64_t now_ns) const {
    const auto it = pairs_.find(pair_key(trader_id, symbol_id));
    if (it == pairs_.end()) {
        return {};
    }
    return window == Window::OneSecond
               ? summarize(it->second.short_window, kShortBucketNs, now_ns)
               : summarize(it->second.long_window, kLongBucketNs, now_ns);
}

}  // namespace tsa
