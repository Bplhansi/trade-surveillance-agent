#pragma once

// Short-term memory of each trader's activity on each symbol.
//
// Two windows per (trader, symbol) pair:
//   OneSecond:    10 buckets x 100 ms
//   SixtySeconds: 60 buckets x 1 s
// Buckets live in a fixed-size ring buffer, so memory per pair never grows.
// A window holds the current bucket plus the ones before it, so its edge moves in
// whole-bucket steps: an event leaves the 1 s window exactly 1 s after the start of
// its 100 ms bucket.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "tsa/event.hpp"
#include "tsa/ring_buffer.hpp"

namespace tsa {

enum class Window { OneSecond, SixtySeconds };

// Orders of at least this many shares count as "large" unless configured otherwise.
inline constexpr std::uint32_t kDefaultLargeOrderQty = 1000;

struct WindowStats {
    std::uint32_t orders = 0;          // New orders placed
    std::uint32_t cancels = 0;         // orders cancelled
    double cancel_ratio = 0.0;         // cancels / orders (0 if no orders)
    double avg_order_qty = 0.0;        // average size of New orders
    std::uint32_t max_order_qty = 0;   // largest New order
    std::uint32_t large_orders = 0;    // New orders of at least the "large" size
    std::uint32_t large_cancels = 0;   // cancels of orders of at least the "large" size
    double large_cancel_ratio = 0.0;   // large_cancels / large_orders (0 if none placed)
    std::uint64_t buy_qty = 0;         // shares bought (own Trade events, buy side)
    std::uint64_t sell_qty = 0;        // shares sold
    std::int64_t net_qty = 0;          // buy_qty - sell_qty: change in position
    std::uint32_t trades = 0;          // own Trade events
    double first_trade_price = 0.0;    // price of the first own trade in the window (0 if none)
    double price_change = 0.0;         // last own trade price - first own trade price
    std::uint32_t self_trades = 0;     // fills where this trader was both buyer and seller
};

class RollingStats {
public:
    explicit RollingStats(std::uint32_t large_order_qty = kDefaultLargeOrderQty)
        : large_order_qty_(large_order_qty) {}

    static constexpr std::uint64_t kShortBucketNs = 100'000'000;  // 100 ms
    static constexpr std::size_t kShortBuckets = 10;               // -> 1 s window
    static constexpr std::uint64_t kLongBucketNs = 1'000'000'000;  // 1 s
    static constexpr std::size_t kLongBuckets = 60;                // -> 60 s window

    // Events must arrive in time order (EventReader guarantees this).
    void update(const Event& e);

    // Statistics for one trader on one symbol, as of time now_ns.
    WindowStats stats(std::uint32_t trader_id, std::uint32_t symbol_id, Window window,
                      std::uint64_t now_ns) const;
    // Same, as of the latest event seen.
    WindowStats stats(std::uint32_t trader_id, std::uint32_t symbol_id, Window window) const {
        return stats(trader_id, symbol_id, window, last_ts_ns_);
    }

    std::size_t tracked_pairs() const noexcept { return pairs_.size(); }

private:
    struct Bucket {
        std::uint32_t orders = 0;
        std::uint32_t cancels = 0;
        std::uint32_t trades = 0;
        std::uint32_t self_trades = 0;
        std::uint32_t max_order_qty = 0;
        std::uint32_t large_orders = 0;
        std::uint32_t large_cancels = 0;
        std::uint64_t order_qty = 0;
        std::uint64_t buy_qty = 0;
        std::uint64_t sell_qty = 0;
        double first_price = 0.0;
        double last_price = 0.0;
    };
    struct PairWindows {
        RingBuffer<Bucket, kShortBuckets> short_window;
        RingBuffer<Bucket, kLongBuckets> long_window;
    };

    void apply(Bucket& b, const Event& e, bool self_trade) const;
    template <std::size_t N>
    static WindowStats summarize(const RingBuffer<Bucket, N>& ring, std::uint64_t bucket_ns,
                                 std::uint64_t now_ns);

    std::uint32_t large_order_qty_;
    std::unordered_map<std::uint64_t, PairWindows> pairs_;  // key: trader << 32 | symbol
    // Trade halves at the current timestamp still waiting for their other side. Both halves of
    // a fill share one timestamp, so this is cleared whenever time moves on (capacity is kept,
    // so it stops allocating after warm-up).
    std::vector<Event> unmatched_trades_;
    std::uint64_t last_ts_ns_ = 0;
};

}  // namespace tsa
