#include "tsa/market_state.hpp"

#include <cstddef>

namespace tsa {

namespace {
const SymbolStats kNoStats{};
}  // namespace

void MarketState::update(const Event& e) {
    if (!is_valid(e)) {
        return;  // EventReader rejects these; this guards direct callers too
    }
    ++counts_[static_cast<std::size_t>(e.type)];

    switch (e.type) {
        case EventType::New:
            open_.insert(e.order_id);
            break;
        case EventType::Cancel:
            open_.erase(e.order_id);
            break;
        case EventType::Trade: {
            open_.erase(e.order_id);
            if (e.symbol_id >= symbols_.size()) {
                symbols_.resize(std::size_t{e.symbol_id} + 1);
            }
            SymbolStats& s = symbols_[e.symbol_id];
            s.last_price = e.price;
            // Every fill appears as two Trade events, one per side, so count the buy side
            // only. This also counts a trader's buy-from-self (two order IDs) once.
            if (e.side == Side::Buy) {
                s.traded_qty += e.quantity;
                ++s.fills;
            }
            break;
        }
    }
}

const SymbolStats& MarketState::symbol(std::uint32_t symbol_id) const noexcept {
    return symbol_id < symbols_.size() ? symbols_[symbol_id] : kNoStats;
}

}  // namespace tsa
