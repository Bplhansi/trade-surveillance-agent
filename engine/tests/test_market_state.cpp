// Tests for the engine's small market state (counts, prices, volume, open orders).
#include <gtest/gtest.h>

#include "test_helpers.hpp"
#include "tsa/market_state.hpp"

using tsa::EventType;
using tsa::Side;
using tsa_test::make_event;

TEST(MarketState, StartsEmpty) {
    const tsa::MarketState state;

    EXPECT_EQ(state.count(EventType::New), 0u);
    EXPECT_EQ(state.open_orders(), 0u);
    EXPECT_EQ(state.symbol(1).last_price, 0.0);
    EXPECT_EQ(state.symbol(1).traded_qty, 0u);
}

TEST(MarketState, NewOrderStaysOpenUntilCancelled) {
    tsa::MarketState state;

    state.update(make_event(1, 5, 7, 1, 10.0, 100, EventType::New, Side::Buy));
    EXPECT_TRUE(state.is_open(5));
    EXPECT_EQ(state.open_orders(), 1u);

    state.update(make_event(2, 5, 7, 1, 10.0, 100, EventType::Cancel, Side::Buy));
    EXPECT_FALSE(state.is_open(5));
    EXPECT_EQ(state.open_orders(), 0u);
}

TEST(MarketState, FillClosesTheOrderAndCountsVolumeOnce) {
    // A fill is two Trade events with the same order_id: the owner and the counterparty.
    tsa::MarketState state;
    state.update(make_event(1, 5, 7, 1, 10.5, 100, EventType::New, Side::Buy));
    state.update(make_event(2, 5, 7, 1, 10.5, 100, EventType::Trade, Side::Buy));
    state.update(make_event(2, 5, 8, 1, 10.5, 100, EventType::Trade, Side::Sell));

    EXPECT_FALSE(state.is_open(5));
    EXPECT_EQ(state.symbol(1).last_price, 10.5);
    EXPECT_EQ(state.symbol(1).traded_qty, 100u);
    EXPECT_EQ(state.symbol(1).fills, 1u);
    EXPECT_EQ(state.count(EventType::Trade), 2u);
}

TEST(MarketState, SelfTradeBetweenTwoOwnOrdersCountsVolumeOnce) {
    tsa::MarketState state;
    state.update(make_event(1, 1, 7, 1, 20.0, 300, EventType::New, Side::Buy));
    state.update(make_event(2, 2, 7, 1, 20.0, 300, EventType::New, Side::Sell));
    state.update(make_event(3, 1, 7, 1, 20.0, 300, EventType::Trade, Side::Buy));
    state.update(make_event(3, 2, 7, 1, 20.0, 300, EventType::Trade, Side::Sell));

    EXPECT_EQ(state.open_orders(), 0u);
    EXPECT_EQ(state.symbol(1).traded_qty, 300u);
}

TEST(MarketState, CountsEventsByType) {
    tsa::MarketState state;
    state.update(make_event(1, 1, 1, 1, 1.0, 10, EventType::New, Side::Buy));
    state.update(make_event(2, 2, 1, 1, 1.0, 10, EventType::New, Side::Sell));
    state.update(make_event(3, 1, 1, 1, 1.0, 10, EventType::Cancel, Side::Buy));

    EXPECT_EQ(state.count(EventType::New), 2u);
    EXPECT_EQ(state.count(EventType::Cancel), 1u);
    EXPECT_EQ(state.count(EventType::Trade), 0u);
}

TEST(MarketState, KeepsSymbolsSeparate) {
    tsa::MarketState state;
    state.update(make_event(1, 1, 1, 1, 10.0, 50, EventType::Trade, Side::Buy));
    state.update(make_event(2, 2, 1, 3, 99.0, 70, EventType::Trade, Side::Buy));

    EXPECT_EQ(state.symbol(1).last_price, 10.0);
    EXPECT_EQ(state.symbol(3).last_price, 99.0);
    EXPECT_EQ(state.symbol(2).last_price, 0.0);  // never traded
}

TEST(MarketState, IgnoresInvalidEventsInsteadOfCorruptingMemory) {
    tsa::MarketState state;
    auto bad_type = make_event(1, 1, 1, 1, 1.0, 10, EventType::New, Side::Buy);
    bad_type.type = static_cast<EventType>(7);
    auto huge_symbol = make_event(2, 2, 1, 0xFFFFFFFF, 1.0, 10, EventType::Trade, Side::Buy);

    state.update(bad_type);
    state.update(huge_symbol);

    EXPECT_EQ(state.count(EventType::New) + state.count(EventType::Cancel) +
                  state.count(EventType::Trade),
              0u);
    EXPECT_EQ(state.open_orders(), 0u);
}

TEST(MarketState, CancelOfAnUnknownOrderIsIgnored) {
    tsa::MarketState state;

    state.update(make_event(1, 42, 1, 1, 1.0, 10, EventType::Cancel, Side::Buy));

    EXPECT_EQ(state.open_orders(), 0u);
    EXPECT_EQ(state.count(EventType::Cancel), 1u);
}
