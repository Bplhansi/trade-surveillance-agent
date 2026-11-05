// Tests for the .ini config reader and the rule thresholds it fills in.
#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "test_helpers.hpp"
#include "tsa/config.hpp"
#include "tsa/rules.hpp"

namespace {

// EXPECT_THROW plus a check that the message mentions `needle`.
template <class F>
void expect_error_mentioning(F&& f, const std::string& needle) {
    try {
        f();
        FAIL() << "expected an exception mentioning '" << needle << "'";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
    }
}

}  // namespace

TEST(Config, ParsesSectionsKeysAndComments) {
    const auto cfg = tsa::Config::from_string(
        "; a comment\n"
        "[spoofing]\n"
        "large_order_qty = 1500   ; trailing comment\n"
        "  min_large_cancel_ratio=0.75\n"
        "\n"
        "# another comment\n"
        "[general]\n"
        "cooldown_s = 30\n");

    EXPECT_DOUBLE_EQ(cfg.get_double("spoofing", "large_order_qty", 0), 1500);
    EXPECT_DOUBLE_EQ(cfg.get_double("spoofing", "min_large_cancel_ratio", 0), 0.75);
    EXPECT_DOUBLE_EQ(cfg.get_double("general", "cooldown_s", 0), 30);
    EXPECT_TRUE(cfg.has("general", "cooldown_s"));
}

TEST(Config, MissingKeyUsesTheFallback) {
    const auto cfg = tsa::Config::from_string("[spoofing]\n");

    EXPECT_FALSE(cfg.has("spoofing", "large_order_qty"));
    EXPECT_DOUBLE_EQ(cfg.get_double("spoofing", "large_order_qty", 1234), 1234);
}

TEST(Config, RejectsALineWithoutEqualsSign) {
    expect_error_mentioning([] { tsa::Config::from_string("[a]\nnot a setting\n", "x.ini"); },
                            "x.ini:2");
}

TEST(Config, RejectsAKeyOutsideAnySection) {
    expect_error_mentioning([] { tsa::Config::from_string("key = 1\n", "x.ini"); }, "section");
}

TEST(Config, RejectsAValueThatIsNotANumber) {
    const auto cfg = tsa::Config::from_string("[a]\nb = 12abc\n", "x.ini");

    expect_error_mentioning([&] { cfg.get_double("a", "b", 0); }, "a.b");
}

TEST(Config, FromFileThrowsWhenTheFileIsMissing) {
    EXPECT_THROW(tsa::Config::from_file(tsa_test::temp_path("no_such_rules.ini")),
                 std::runtime_error);
}

TEST(RulesConfig, ReadsEveryThreshold) {
    const auto cfg = tsa::RulesConfig::from(tsa::Config::from_string(
        "[general]\ncooldown_s = 30\n"
        "[spoofing]\nlarge_order_qty = 1500\nmin_large_cancels = 4\nmin_large_cancel_ratio = 0.9\n"
        "[wash_trade]\nmin_trades = 8\nmin_volume = 700\nmax_net_ratio = 0.1\nmin_self_trades = 2\n"
        "[pump_and_dump]\nmin_net_qty = 5000\nmin_rise_pct = 2.5\nmin_sell_qty = 900\n"
        "sell_within_s = 300\n"));

    EXPECT_DOUBLE_EQ(cfg.cooldown_s, 30);
    EXPECT_EQ(cfg.spoofing.large_order_qty, 1500u);
    EXPECT_EQ(cfg.spoofing.min_large_cancels, 4u);
    EXPECT_DOUBLE_EQ(cfg.spoofing.min_large_cancel_ratio, 0.9);
    EXPECT_EQ(cfg.wash_trade.min_trades, 8u);
    EXPECT_EQ(cfg.wash_trade.min_volume, 700u);
    EXPECT_DOUBLE_EQ(cfg.wash_trade.max_net_ratio, 0.1);
    EXPECT_EQ(cfg.wash_trade.min_self_trades, 2u);
    EXPECT_EQ(cfg.pump_and_dump.min_net_qty, 5000);
    EXPECT_DOUBLE_EQ(cfg.pump_and_dump.min_rise_pct, 2.5);
    EXPECT_EQ(cfg.pump_and_dump.min_sell_qty, 900u);
    EXPECT_DOUBLE_EQ(cfg.pump_and_dump.sell_within_s, 300);
}

TEST(RulesConfig, UsesDefaultsForMissingKeys) {
    const auto cfg = tsa::RulesConfig::from(tsa::Config::from_string(""));
    const tsa::RulesConfig defaults;

    EXPECT_EQ(cfg.spoofing.large_order_qty, defaults.spoofing.large_order_qty);
    EXPECT_DOUBLE_EQ(cfg.cooldown_s, defaults.cooldown_s);
}

TEST(RulesConfig, RejectsAMisspelledKey) {
    // A typo must not silently fall back to the default threshold.
    expect_error_mentioning(
        [] { tsa::RulesConfig::from(tsa::Config::from_string("[spoofing]\nlarge_ordr_qty = 5\n")); },
        "spoofing.large_ordr_qty");
}

TEST(RulesConfig, RejectsNegativeOrOutOfRangeThresholds) {
    EXPECT_THROW(
        tsa::RulesConfig::from(tsa::Config::from_string("[spoofing]\nlarge_order_qty = -1\n")),
        std::runtime_error);
    EXPECT_THROW(
        tsa::RulesConfig::from(tsa::Config::from_string("[wash_trade]\nmax_net_ratio = 1.5\n")),
        std::runtime_error);
}

TEST(RulesConfig, TheProjectRulesFileLoads) {
    EXPECT_NO_THROW(tsa::RulesConfig::from(tsa::Config::from_file(TSA_RULES_FILE)));
}
