// Tests for the three baseline rules: each with a case that should fire and one that shouldn't.
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "test_helpers.hpp"
#include "tsa/engine_state.hpp"
#include "tsa/rules.hpp"

using tsa::EventType;
using tsa::Rule;
using tsa::Side;
using tsa_test::make_event;

namespace {

constexpr std::uint64_t kMs = 1'000'000;
constexpr std::uint64_t kSec = 1'000 * kMs;
constexpr std::uint64_t kT0 = 1'000 * kSec;

tsa::RulesConfig test_config() {
    tsa::RulesConfig cfg;
    cfg.cooldown_s = 60;
    cfg.spoofing = {.large_order_qty = 1000, .min_large_cancels = 3, .min_large_cancel_ratio = 0.8};
    cfg.wash_trade = {.min_trades = 4, .min_volume = 100, .max_net_ratio = 0.05};
    cfg.pump_and_dump = {.min_net_qty = 300, .min_rise_pct = 1.0, .min_sell_qty = 100,
                         .sell_within_s = 600};
    return cfg;
}

// Runs events through state + rules the way the replay loop does; returns the alerts.
std::vector<tsa::Alert> run(const std::vector<tsa::Event>& events,
                            const tsa::RulesConfig& cfg = test_config()) {
    tsa::EngineState state(cfg.spoofing.large_order_qty);
    tsa::Rules rules(cfg);
    std::vector<tsa::Alert> alerts;
    for (const auto& e : events) {
        state.update(e);
        if (auto alert = rules.check(e, state)) {
            alerts.push_back(*alert);
        }
    }
    return alerts;
}

// Trader 7 places `n` large buy orders on symbol 1, then cancels them all.
std::vector<tsa::Event> spoof_burst(std::uint64_t start, std::uint32_t first_order, int n,
                                    std::uint32_t qty = 2000) {
    std::vector<tsa::Event> events;
    for (int i = 0; i < n; ++i) {
        events.push_back(make_event(start + i * 2 * kMs, first_order + i, 7, 1, 10.0 - 0.01 * i,
                                    qty, EventType::New, Side::Buy));
    }
    for (int i = 0; i < n; ++i) {
        events.push_back(make_event(start + 100 * kMs + i * 10 * kMs, first_order + i, 7, 1,
                                    10.0 - 0.01 * i, qty, EventType::Cancel, Side::Buy));
    }
    return events;
}

tsa::Event trade(std::uint64_t ts, std::uint32_t order, std::uint32_t trader, double price,
                 std::uint32_t qty, Side side) {
    return make_event(ts, order, trader, 1, price, qty, EventType::Trade, side);
}

}  // namespace

// --- Spoofing ---------------------------------------------------------------------

TEST(SpoofingRule, FiresWhenSeveralLargeOrdersAreCancelledQuickly) {
    const auto alerts = run(spoof_burst(kT0, 1, 3));

    ASSERT_EQ(alerts.size(), 1u);
    EXPECT_EQ(alerts[0].rule, Rule::Spoofing);
    EXPECT_EQ(alerts[0].trader_id, 7u);
    EXPECT_EQ(alerts[0].symbol_id, 1u);
    EXPECT_EQ(alerts[0].event_index, 5u);  // the third cancel is the 6th event
    EXPECT_EQ(alerts[0].evidence.large_cancels, 3u);
    EXPECT_DOUBLE_EQ(alerts[0].evidence.large_cancel_ratio, 1.0);
    EXPECT_EQ(alerts[0].window, tsa::Window::OneSecond);
}

TEST(SpoofingRule, IgnoresASingleLargeCancelledOrder) {
    // An innocent "decoy": one big order, cancelled.
    EXPECT_TRUE(run(spoof_burst(kT0, 1, 1)).empty());
}

TEST(SpoofingRule, IgnoresSmallOrdersBeingCancelled) {
    EXPECT_TRUE(run(spoof_burst(kT0, 1, 5, /*qty=*/100)).empty());
}

TEST(SpoofingRule, IgnoresLargeOrdersThatMostlyStayOpen) {
    auto events = spoof_burst(kT0, 1, 3);
    // Place 2 more large orders that are never cancelled: 3 cancels / 5 large = 0.6 < 0.8.
    events.insert(events.begin() + 3,
                  make_event(kT0 + 6 * kMs, 50, 7, 1, 9.9, 2000, EventType::New, Side::Buy));
    events.insert(events.begin() + 4,
                  make_event(kT0 + 8 * kMs, 51, 7, 1, 9.9, 2000, EventType::New, Side::Buy));

    EXPECT_TRUE(run(events).empty());
}

// --- Wash trade -------------------------------------------------------------------

TEST(WashTradeRule, FiresOnBalancedBuyingAndSellingByOneAccount) {
    const std::vector<tsa::Event> events = {
        trade(kT0, 1, 5, 20.0, 100, Side::Buy),
        trade(kT0, 2, 5, 20.0, 100, Side::Sell),
        trade(kT0 + kSec, 3, 5, 20.0, 100, Side::Buy),
        trade(kT0 + kSec, 4, 5, 20.0, 100, Side::Sell),
    };

    const auto alerts = run(events);

    ASSERT_EQ(alerts.size(), 1u);
    EXPECT_EQ(alerts[0].rule, Rule::WashTrade);
    EXPECT_EQ(alerts[0].trader_id, 5u);
    EXPECT_EQ(alerts[0].event_index, 3u);
    EXPECT_EQ(alerts[0].evidence.net_qty, 0);
    EXPECT_EQ(alerts[0].window, tsa::Window::SixtySeconds);
}

TEST(WashTradeRule, WithMinSelfTradesIgnoresBalancedTradingWithOthers) {
    auto cfg = test_config();
    cfg.wash_trade.min_self_trades = 1;
    // Balanced buying and selling, but always against trader 6: not a wash trade.
    std::vector<tsa::Event> events;
    for (std::uint32_t i = 0; i < 2; ++i) {
        const auto ts = kT0 + i * kSec;
        events.push_back(trade(ts, 10 * i + 1, 5, 20.0, 100, Side::Buy));
        events.push_back(trade(ts, 10 * i + 2, 6, 20.0, 100, Side::Sell));
        events.push_back(trade(ts + kMs, 10 * i + 3, 5, 20.0, 100, Side::Sell));
        events.push_back(trade(ts + kMs, 10 * i + 4, 6, 20.0, 100, Side::Buy));
    }

    EXPECT_TRUE(run(events, cfg).empty());
}

TEST(WashTradeRule, WithMinSelfTradesStillFiresWhenTheAccountTradesWithItself) {
    auto cfg = test_config();
    cfg.wash_trade.min_self_trades = 1;
    const std::vector<tsa::Event> events = {
        trade(kT0, 1, 5, 20.0, 100, Side::Buy),
        trade(kT0, 2, 5, 20.0, 100, Side::Sell),  // same fill: a self-trade
        trade(kT0 + kSec, 3, 5, 20.0, 100, Side::Buy),
        trade(kT0 + kSec, 4, 5, 20.0, 100, Side::Sell),
    };

    const auto alerts = run(events, cfg);

    ASSERT_EQ(alerts.size(), 1u);
    EXPECT_EQ(alerts[0].evidence.self_trades, 2u);
}

TEST(WashTradeRule, IgnoresAnAccountThatOnlyBuys) {
    std::vector<tsa::Event> events;
    for (std::uint32_t i = 0; i < 6; ++i) {
        events.push_back(trade(kT0 + i * kSec, i + 1, 5, 20.0, 100, Side::Buy));
    }

    EXPECT_TRUE(run(events).empty());
}

TEST(WashTradeRule, IgnoresTooFewTrades) {
    const std::vector<tsa::Event> events = {
        trade(kT0, 1, 5, 20.0, 100, Side::Buy),
        trade(kT0, 2, 5, 20.0, 100, Side::Sell),
    };

    EXPECT_TRUE(run(events).empty());
}

// --- Pump and dump ----------------------------------------------------------------

namespace {

// Trader 9 buys 3 x 100 while the price rises 1.2%, then sells 150 thirty seconds later.
std::vector<tsa::Event> pump_then_sell(std::uint64_t sell_delay) {
    return {
        trade(kT0, 1, 9, 10.00, 100, Side::Buy),
        trade(kT0 + 1 * kSec, 2, 9, 10.05, 100, Side::Buy),
        trade(kT0 + 2 * kSec, 3, 9, 10.12, 100, Side::Buy),
        trade(kT0 + 2 * kSec + sell_delay, 4, 9, 10.10, 150, Side::Sell),
    };
}

}  // namespace

TEST(PumpAndDumpRule, FiresWhenTheAccountThatPushedThePriceUpSells) {
    const auto alerts = run(pump_then_sell(30 * kSec));

    ASSERT_EQ(alerts.size(), 1u);
    EXPECT_EQ(alerts[0].rule, Rule::PumpAndDump);
    EXPECT_EQ(alerts[0].trader_id, 9u);
    EXPECT_EQ(alerts[0].event_index, 3u);
    EXPECT_EQ(alerts[0].evidence.sell_qty, 150u);
}

TEST(PumpAndDumpRule, IgnoresSellingWithoutAPumpFirst) {
    const std::vector<tsa::Event> events = {
        trade(kT0, 1, 9, 10.00, 100, Side::Buy),
        trade(kT0 + kSec, 2, 9, 10.01, 100, Side::Buy),  // rise far below 1%
        trade(kT0 + 2 * kSec, 3, 9, 10.01, 100, Side::Buy),
        trade(kT0 + 30 * kSec, 4, 9, 10.00, 150, Side::Sell),
    };

    EXPECT_TRUE(run(events).empty());
}

TEST(PumpAndDumpRule, IgnoresSellingLongAfterThePump) {
    EXPECT_TRUE(run(pump_then_sell(700 * kSec)).empty());  // sell_within_s is 600
}

TEST(PumpAndDumpRule, IgnoresADifferentAccountSelling) {
    auto events = pump_then_sell(30 * kSec);
    events.back().trader_id = 10;

    EXPECT_TRUE(run(events).empty());
}

// --- Shared behaviour ---------------------------------------------------------------

TEST(Rules, CooldownLimitsOneCaseToOneAlert) {
    auto events = spoof_burst(kT0, 1, 3);
    const auto second = spoof_burst(kT0 + 5 * kSec, 10, 3);   // within the 60 s cooldown
    const auto third = spoof_burst(kT0 + 70 * kSec, 20, 3);   // after it
    events.insert(events.end(), second.begin(), second.end());
    events.insert(events.end(), third.begin(), third.end());

    const auto alerts = run(events);

    ASSERT_EQ(alerts.size(), 2u);
    EXPECT_EQ(alerts[1].timestamp_ns, kT0 + 70 * kSec + 120 * kMs);
}

TEST(Rules, RuleNamesMatchThePythonLabels) {
    EXPECT_EQ(tsa::rule_name(Rule::Spoofing), "spoofing");
    EXPECT_EQ(tsa::rule_name(Rule::WashTrade), "wash_trade");
    EXPECT_EQ(tsa::rule_name(Rule::PumpAndDump), "pump_and_dump");
}
