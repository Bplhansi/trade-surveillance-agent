// Tests for per-trader, per-symbol rolling statistics over 1 s and 60 s windows.
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "test_helpers.hpp"
#include "tsa/engine_state.hpp"
#include "tsa/rolling_stats.hpp"

using tsa::EventType;
using tsa::Side;
using tsa::Window;
using tsa_test::make_event;

namespace {

constexpr std::uint64_t kMs = 1'000'000;
constexpr std::uint64_t kStart = 1'000'000 * kMs;  // a whole number of seconds

// One trader (7) on symbol 1, with innocent counterparties 8 and 9:
//   0 ms  New buy  100 @ 10.00 (order 1)
//  50 ms  New buy  300 @ 10.01 (order 2)
// 120 ms  Cancel order 2
// 200 ms  order 1 fills: trader 7 buys 100 @ 10.00 from trader 8
// 400 ms  New sell 200 @ 10.05 (order 3)
// 450 ms  order 3 fills: trader 7 sells 200 @ 10.05 to trader 9
// 600 ms  New buy  50 @ 10.10 (order 4)
// 601 ms  New sell 50 @ 10.10 (order 5)
// 602 ms  orders 4 and 5 fill against each other: a self-trade by trader 7
std::vector<tsa::Event> known_sequence() {
    return {
        make_event(kStart + 0 * kMs, 1, 7, 1, 10.00, 100, EventType::New, Side::Buy),
        make_event(kStart + 50 * kMs, 2, 7, 1, 10.01, 300, EventType::New, Side::Buy),
        make_event(kStart + 120 * kMs, 2, 7, 1, 10.01, 300, EventType::Cancel, Side::Buy),
        make_event(kStart + 200 * kMs, 1, 7, 1, 10.00, 100, EventType::Trade, Side::Buy),
        make_event(kStart + 200 * kMs, 1, 8, 1, 10.00, 100, EventType::Trade, Side::Sell),
        make_event(kStart + 400 * kMs, 3, 7, 1, 10.05, 200, EventType::New, Side::Sell),
        make_event(kStart + 450 * kMs, 3, 7, 1, 10.05, 200, EventType::Trade, Side::Sell),
        make_event(kStart + 450 * kMs, 3, 9, 1, 10.05, 200, EventType::Trade, Side::Buy),
        make_event(kStart + 600 * kMs, 4, 7, 1, 10.10, 50, EventType::New, Side::Buy),
        make_event(kStart + 601 * kMs, 5, 7, 1, 10.10, 50, EventType::New, Side::Sell),
        make_event(kStart + 602 * kMs, 4, 7, 1, 10.10, 50, EventType::Trade, Side::Buy),
        make_event(kStart + 602 * kMs, 5, 7, 1, 10.10, 50, EventType::Trade, Side::Sell),
    };
}

tsa::RollingStats fed_with(const std::vector<tsa::Event>& events) {
    tsa::RollingStats stats;
    for (const auto& e : events) {
        stats.update(e);
    }
    return stats;
}

}  // namespace

TEST(RollingStats, ExactStatisticsForTheKnownSequence) {
    const auto stats = fed_with(known_sequence());

    const auto s = stats.stats(7, 1, Window::OneSecond, kStart + 602 * kMs);
    EXPECT_EQ(s.orders, 5u);
    EXPECT_EQ(s.cancels, 1u);
    EXPECT_DOUBLE_EQ(s.cancel_ratio, 0.2);
    EXPECT_DOUBLE_EQ(s.avg_order_qty, 140.0);  // (100 + 300 + 200 + 50 + 50) / 5
    EXPECT_EQ(s.max_order_qty, 300u);
    EXPECT_EQ(s.buy_qty, 150u);   // 100 + 50
    EXPECT_EQ(s.sell_qty, 250u);  // 200 + 50
    EXPECT_EQ(s.net_qty, -100);   // bought 150, sold 250
    EXPECT_EQ(s.trades, 4u);      // trader 7's own Trade events
    EXPECT_NEAR(s.price_change, 0.10, 1e-9);  // first trade 10.00, last trade 10.10
    EXPECT_EQ(s.self_trades, 1u);
}

TEST(RollingStats, SixtySecondWindowMatchesWhileEverythingIsRecent) {
    const auto stats = fed_with(known_sequence());

    const auto one = stats.stats(7, 1, Window::OneSecond, kStart + 602 * kMs);
    const auto sixty = stats.stats(7, 1, Window::SixtySeconds, kStart + 602 * kMs);
    EXPECT_EQ(sixty.orders, one.orders);
    EXPECT_EQ(sixty.buy_qty, one.buy_qty);
    EXPECT_EQ(sixty.self_trades, one.self_trades);
}

TEST(RollingStats, OneSecondWindowDropsOldBucketsAsTimeMovesOn) {
    const auto stats = fed_with(known_sequence());

    // At 1.5 s only the 600-699 ms bucket is still inside the last second.
    const auto s = stats.stats(7, 1, Window::OneSecond, kStart + 1500 * kMs);
    EXPECT_EQ(s.orders, 2u);
    EXPECT_EQ(s.cancels, 0u);
    EXPECT_DOUBLE_EQ(s.cancel_ratio, 0.0);
    EXPECT_DOUBLE_EQ(s.avg_order_qty, 50.0);
    EXPECT_EQ(s.buy_qty, 50u);
    EXPECT_EQ(s.sell_qty, 50u);
    EXPECT_EQ(s.net_qty, 0);
    EXPECT_EQ(s.trades, 2u);
    EXPECT_DOUBLE_EQ(s.price_change, 0.0);
    EXPECT_EQ(s.self_trades, 1u);

    // At 1.7 s nothing is left in the last second, but all of it is in the last minute.
    EXPECT_EQ(stats.stats(7, 1, Window::OneSecond, kStart + 1700 * kMs).orders, 0u);
    EXPECT_EQ(stats.stats(7, 1, Window::SixtySeconds, kStart + 1700 * kMs).orders, 5u);
}

TEST(RollingStats, EventLeavesTheWindowExactlyOneWindowLater) {
    tsa::RollingStats stats;
    stats.update(make_event(kStart, 1, 7, 1, 10.0, 100, EventType::New, Side::Buy));

    EXPECT_EQ(stats.stats(7, 1, Window::OneSecond, kStart + 999 * kMs).orders, 1u);
    EXPECT_EQ(stats.stats(7, 1, Window::OneSecond, kStart + 1000 * kMs).orders, 0u);
    EXPECT_EQ(stats.stats(7, 1, Window::SixtySeconds, kStart + 59'999 * kMs).orders, 1u);
    EXPECT_EQ(stats.stats(7, 1, Window::SixtySeconds, kStart + 60'000 * kMs).orders, 0u);
}

TEST(RollingStats, CounterpartiesGetTheirOwnStatistics) {
    const auto stats = fed_with(known_sequence());

    const auto s8 = stats.stats(8, 1, Window::OneSecond, kStart + 602 * kMs);
    EXPECT_EQ(s8.orders, 0u);
    EXPECT_EQ(s8.trades, 1u);
    EXPECT_EQ(s8.sell_qty, 100u);
    EXPECT_EQ(s8.net_qty, -100);
    EXPECT_EQ(s8.self_trades, 0u);  // traded with someone else
}

TEST(RollingStats, KeepsSymbolsSeparate) {
    tsa::RollingStats stats;
    stats.update(make_event(kStart, 1, 7, 1, 10.0, 100, EventType::New, Side::Buy));
    stats.update(make_event(kStart + kMs, 2, 7, 2, 20.0, 500, EventType::New, Side::Buy));

    EXPECT_EQ(stats.stats(7, 1, Window::OneSecond, kStart + kMs).max_order_qty, 100u);
    EXPECT_EQ(stats.stats(7, 2, Window::OneSecond, kStart + kMs).max_order_qty, 500u);
    EXPECT_EQ(stats.tracked_pairs(), 2u);
}

TEST(RollingStats, UnknownTraderHasAllZeroStatistics) {
    const auto stats = fed_with(known_sequence());

    const auto s = stats.stats(123, 1, Window::SixtySeconds, kStart + 602 * kMs);
    EXPECT_EQ(s.orders, 0u);
    EXPECT_EQ(s.trades, 0u);
    EXPECT_DOUBLE_EQ(s.cancel_ratio, 0.0);
    EXPECT_DOUBLE_EQ(s.avg_order_qty, 0.0);
}

TEST(RollingStats, DefaultQueryTimeIsTheLatestEvent) {
    const auto stats = fed_with(known_sequence());

    EXPECT_EQ(stats.stats(7, 1, Window::OneSecond).orders,
              stats.stats(7, 1, Window::OneSecond, kStart + 602 * kMs).orders);
}

TEST(RollingStats, MemoryPerTraderSymbolPairDoesNotGrowWithEvents) {
    tsa::RollingStats stats;
    for (std::uint32_t i = 0; i < 100'000; ++i) {
        stats.update(make_event(kStart + i * kMs, i + 1, 7, 1, 10.0, 100, EventType::New, Side::Buy));
    }

    EXPECT_EQ(stats.tracked_pairs(), 1u);
    // Only the last 60 s (60,000 events at 1 per ms) are still counted.
    EXPECT_EQ(stats.stats(7, 1, Window::SixtySeconds).orders, 60'000u);
    EXPECT_EQ(stats.stats(7, 1, Window::OneSecond).orders, 1'000u);
}

TEST(RollingStats, PairsASelfTradeEvenWhenAnotherFillArrivesBetweenItsHalves) {
    tsa::RollingStats stats;
    const auto t = kStart;
    stats.update(make_event(t, 1, 7, 1, 10.0, 100, EventType::Trade, Side::Buy));   // 7 buys...
    stats.update(make_event(t, 2, 3, 2, 55.0, 300, EventType::Trade, Side::Buy));   // other fill
    stats.update(make_event(t, 3, 4, 2, 55.0, 300, EventType::Trade, Side::Sell));
    stats.update(make_event(t, 4, 7, 1, 10.0, 100, EventType::Trade, Side::Sell));  // ...from 7

    EXPECT_EQ(stats.stats(7, 1, Window::OneSecond).self_trades, 1u);
    EXPECT_EQ(stats.stats(3, 2, Window::OneSecond).self_trades, 0u);
}

TEST(RollingStats, DoesNotPairTradesFromDifferentMoments) {
    tsa::RollingStats stats;
    stats.update(make_event(kStart, 1, 7, 1, 10.0, 100, EventType::Trade, Side::Buy));
    stats.update(make_event(kStart + 1, 2, 7, 1, 10.0, 100, EventType::Trade, Side::Sell));

    EXPECT_EQ(stats.stats(7, 1, Window::OneSecond).self_trades, 0u);
}

TEST(RollingStats, AFillBetweenTwoDifferentTradersIsNotASelfTrade) {
    tsa::RollingStats stats;
    stats.update(make_event(kStart, 1, 7, 1, 10.0, 100, EventType::Trade, Side::Buy));
    stats.update(make_event(kStart, 2, 8, 1, 10.0, 100, EventType::Trade, Side::Sell));
    stats.update(make_event(kStart, 3, 7, 1, 10.0, 100, EventType::Trade, Side::Sell));

    // 7's buy was already paired with 8's sell; 7's later sell has no partner yet.
    EXPECT_EQ(stats.stats(7, 1, Window::OneSecond).self_trades, 0u);
}

TEST(RollingStats, CountsLargeOrdersAndCancelsFromTheThresholdUp) {
    tsa::RollingStats stats(/*large_order_qty=*/1000);
    stats.update(make_event(kStart, 1, 7, 1, 10.0, 999, EventType::New, Side::Buy));
    stats.update(make_event(kStart + kMs, 2, 7, 1, 10.0, 1000, EventType::New, Side::Buy));
    stats.update(make_event(kStart + 2 * kMs, 2, 7, 1, 10.0, 1000, EventType::Cancel, Side::Buy));
    stats.update(make_event(kStart + 3 * kMs, 1, 7, 1, 10.0, 999, EventType::Cancel, Side::Buy));

    const auto s = stats.stats(7, 1, Window::OneSecond);
    EXPECT_EQ(s.large_orders, 1u);
    EXPECT_EQ(s.large_cancels, 1u);
    EXPECT_DOUBLE_EQ(s.large_cancel_ratio, 1.0);
    EXPECT_EQ(s.cancels, 2u);
}

TEST(RollingStats, RemembersTheFirstTradePriceInTheWindow) {
    const auto stats = fed_with(known_sequence());

    EXPECT_DOUBLE_EQ(stats.stats(7, 1, Window::OneSecond, kStart + 602 * kMs).first_trade_price,
                     10.00);
}

TEST(EngineState, UpdatesBothTheMarketStateAndTheRollingStats) {
    tsa::EngineState state;
    for (const auto& e : known_sequence()) {
        state.update(e);
    }

    EXPECT_EQ(state.market.count(EventType::New), 5u);
    EXPECT_EQ(state.rolling.stats(7, 1, Window::OneSecond).self_trades, 1u);
}
