#pragma once

// The engine's memory of the market, updated once per event.
// Deliberately small; rolling statistics live in rolling_stats.hpp.

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "tsa/event.hpp"

namespace tsa {

struct SymbolStats {
    double last_price = 0.0;       // price of the most recent trade
    std::uint64_t traded_qty = 0;  // shares traded (each fill counted once)
    std::uint64_t fills = 0;       // number of fills
};

class MarketState {
public:
    void update(const Event& e);

    std::uint64_t count(EventType type) const noexcept {
        return counts_[static_cast<std::size_t>(type)];
    }
    // Stats for one symbol; all zero if it has never traded.
    const SymbolStats& symbol(std::uint32_t symbol_id) const noexcept;
    std::size_t open_orders() const noexcept { return open_.size(); }
    bool is_open(std::uint32_t order_id) const { return open_.contains(order_id); }

private:
    std::array<std::uint64_t, 3> counts_{};
    std::vector<SymbolStats> symbols_;  // index = symbol_id
    std::unordered_set<std::uint32_t> open_;
};

}  // namespace tsa
