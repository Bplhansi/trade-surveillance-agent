#pragma once

// Everything the engine remembers, updated once per event in the replay loop.

#include <cstdint>

#include "tsa/event.hpp"
#include "tsa/market_state.hpp"
#include "tsa/rolling_stats.hpp"

namespace tsa {

struct EngineState {
    explicit EngineState(std::uint32_t large_order_qty = kDefaultLargeOrderQty)
        : rolling(large_order_qty) {}

    MarketState market;
    RollingStats rolling;

    void update(const Event& e) {
        market.update(e);
        rolling.update(e);
    }
};

}  // namespace tsa
