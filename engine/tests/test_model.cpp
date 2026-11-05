// Tests for scoring events with the ONNX model inside the engine.
//
// Python (ml/export_onnx.py --parity-dir) writes a tiny model, 50 feature rows and the scores
// scikit-learn gives them. The C++ scorer must reproduce those scores.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "test_helpers.hpp"
#include "tsa/detector.hpp"
#include "tsa/engine_state.hpp"
#include "tsa/features.hpp"
#include "tsa/model.hpp"

using tsa::EventType;
using tsa::Side;
using tsa_test::make_event;

namespace {

const std::filesystem::path kParityDir = TSA_PARITY_DIR;

template <class T>
std::vector<T> read_all(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::vector<T> v(std::filesystem::file_size(p) / sizeof(T));
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
    return v;
}

#define SKIP_WITHOUT_ONNX()                                                    \
    if (!tsa::model_support_available()) GTEST_SKIP() << "built without ONNX Runtime"

constexpr std::uint64_t kMs = 1'000'000;
constexpr std::uint64_t kSec = 1'000 * kMs;
constexpr std::uint64_t kT0 = 9'000 * kSec;

tsa::Event trade(std::uint64_t ts, std::uint32_t order, std::uint32_t trader, Side side) {
    return make_event(ts, order, trader, 1, 20.0, 100, EventType::Trade, side);
}

}  // namespace

TEST(ModelScorer, MatchesThePythonScoresRowForRow) {
    SKIP_WITHOUT_ONNX();
    const auto rows = read_all<float>(kParityDir / "parity.rows.bin");
    const auto want_scores = read_all<double>(kParityDir / "parity.scores.bin");
    const auto want_patterns = read_all<std::int32_t>(kParityDir / "parity.patterns.bin");
    ASSERT_EQ(rows.size(), want_scores.size() * tsa::kNumFeatures);

    tsa::ModelScorer scorer(kParityDir / "parity.onnx");
    for (std::size_t i = 0; i < want_scores.size(); ++i) {
        tsa::FeatureRow row{};
        std::copy_n(rows.begin() + static_cast<std::ptrdiff_t>(i * tsa::kNumFeatures),
                    tsa::kNumFeatures, row.begin());
        const auto got = scorer.score(row);
        EXPECT_NEAR(got.score, want_scores[i], 1e-3) << "row " << i;
        EXPECT_EQ(static_cast<int>(got.pattern) + 1, want_patterns[i]) << "row " << i;
    }
}

TEST(ModelScorer, ThrowsForAMissingModelFile) {
    SKIP_WITHOUT_ONNX();
    EXPECT_THROW(tsa::ModelScorer(tsa_test::temp_path("no_such_model.onnx")), std::runtime_error);
}

TEST(Gate, OnlyLetsThroughCancelsAndTradesWithEnoughActivity) {
    tsa::ModelConfig cfg;  // defaults: 1 large cancel in 1 s, or 2 trades in 60 s
    tsa::FeatureRow row{};
    const auto at = [](std::string_view n) { return tsa::feature_index(n); };

    row[at("large_cancels_1s")] = 1;
    row[at("trades_60s")] = 5;
    EXPECT_FALSE(tsa::passes_gate(make_event(kT0, 1, 7, 1, 10, 100, EventType::New, Side::Buy), row, cfg));
    EXPECT_TRUE(tsa::passes_gate(make_event(kT0, 1, 7, 1, 10, 100, EventType::Cancel, Side::Buy), row, cfg));
    EXPECT_TRUE(tsa::passes_gate(make_event(kT0, 1, 7, 1, 10, 100, EventType::Trade, Side::Buy), row, cfg));

    row[at("large_cancels_1s")] = 0;
    row[at("trades_60s")] = 1;
    EXPECT_FALSE(tsa::passes_gate(make_event(kT0, 1, 7, 1, 10, 100, EventType::Cancel, Side::Buy), row, cfg));
    EXPECT_FALSE(tsa::passes_gate(make_event(kT0, 1, 7, 1, 10, 100, EventType::Trade, Side::Buy), row, cfg));
}

TEST(Detector, WithoutAModelBehavesLikeTheRules) {
    tsa::RulesConfig cfg;
    cfg.wash_trade = {.min_trades = 4, .min_volume = 100, .max_net_ratio = 0.05};
    tsa::EngineState state;
    tsa::Detector detector(cfg, nullptr);
    std::vector<tsa::Alert> alerts;
    for (const auto& e : {trade(kT0, 1, 5, Side::Buy), trade(kT0, 2, 5, Side::Sell),
                          trade(kT0 + kSec, 3, 5, Side::Buy), trade(kT0 + kSec, 4, 5, Side::Sell)}) {
        state.update(e);
        if (auto a = detector.check(e, state)) alerts.push_back(*a);
    }

    ASSERT_EQ(alerts.size(), 1u);
    EXPECT_EQ(alerts[0].source, tsa::AlertSource::Rules);
    EXPECT_FALSE(alerts[0].model_score.has_value());
    EXPECT_EQ(detector.scored(), 0u);
}

TEST(Detector, RuleAlertsCarryAModelScoreAndModelAlertsAreMarked) {
    SKIP_WITHOUT_ONNX();
    tsa::RulesConfig cfg;
    cfg.wash_trade = {.min_trades = 4, .min_volume = 100, .max_net_ratio = 0.05};
    cfg.model.threshold = -1e9;  // every gated event is "suspicious" to the model
    tsa::ModelScorer scorer(kParityDir / "parity.onnx");
    tsa::EngineState state;
    tsa::Detector detector(cfg, &scorer);
    const std::vector<tsa::Event> events = {
        make_event(kT0, 9, 5, 1, 20.0, 100, EventType::New, Side::Buy),  // never scored
        trade(kT0, 1, 5, Side::Buy), trade(kT0, 2, 5, Side::Sell),
        trade(kT0 + kSec, 3, 5, Side::Buy), trade(kT0 + kSec, 4, 5, Side::Sell)};
    std::vector<tsa::Alert> alerts;
    for (const auto& e : events) {
        state.update(e);
        if (auto a = detector.check(e, state)) alerts.push_back(*a);
    }

    ASSERT_FALSE(alerts.empty());
    EXPECT_EQ(alerts.front().source, tsa::AlertSource::Model);
    EXPECT_EQ(alerts.front().event_index, 2u);  // 2nd trade: first with 2 trades in 60 s
    EXPECT_TRUE(alerts.front().model_score.has_value());
    EXPECT_EQ(alerts.back().source, tsa::AlertSource::Rules);  // the wash-trade rule fired
    EXPECT_EQ(alerts.back().rule, tsa::Rule::WashTrade);
    ASSERT_TRUE(alerts.back().model_score.has_value());
    EXPECT_TRUE(std::isfinite(*alerts.back().model_score));
    EXPECT_EQ(detector.scored(), 4u - 1u);  // trades 2, 3, 4 pass the gate (1 has 1 trade)
}

TEST(RulesConfig, ReadsTheModelSection) {
    const auto cfg = tsa::RulesConfig::from(tsa::Config::from_string(
        "[model]\nthreshold = 6.5\ngate_min_large_cancels_1s = 2\ngate_min_trades_60s = 3\n"));

    EXPECT_DOUBLE_EQ(cfg.model.threshold, 6.5);
    EXPECT_EQ(cfg.model.gate_min_large_cancels_1s, 2u);
    EXPECT_EQ(cfg.model.gate_min_trades_60s, 3u);
    EXPECT_TRUE(std::isinf(tsa::RulesConfig{}.model.threshold));  // off unless configured
}

// Found in code review: the pattern was packed into the top 2 bits of the same
// 64-bit key as the 32-bit trader ID, so large trader IDs shared a cooldown with others.
TEST(Detector, CooldownKeysNeverCollideAcrossPatternsAndTraders) {
    constexpr std::uint32_t kBig = 5u | (1u << 30);
    EXPECT_NE(tsa::model_cooldown_key(tsa::Rule::WashTrade, 5, 1),
              tsa::model_cooldown_key(tsa::Rule::Spoofing, kBig, 1));
    EXPECT_NE(tsa::model_cooldown_key(tsa::Rule::PumpAndDump, 5, 1),
              tsa::model_cooldown_key(tsa::Rule::Spoofing, 5u | (1u << 31), 1));
    EXPECT_EQ(tsa::model_cooldown_key(tsa::Rule::Spoofing, kBig, 1),
              tsa::model_cooldown_key(tsa::Rule::Spoofing, kBig, 1));
}
