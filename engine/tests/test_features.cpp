// Tests for the per-event feature rows the model is trained on and scored on.
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "test_helpers.hpp"
#include "tsa/engine_state.hpp"
#include "tsa/features.hpp"

using tsa::EventType;
using tsa::Side;
using tsa_test::make_event;

namespace {

constexpr std::uint64_t kMs = 1'000'000;
constexpr std::uint64_t kT0 = 5'000'000'000'000ULL;

float get(const tsa::FeatureRow& row, std::string_view name) {
    return row[tsa::feature_index(name)];
}

}  // namespace

TEST(Features, NamesAreFixedAndUnique) {
    const auto& names = tsa::feature_names();
    ASSERT_EQ(names.size(), tsa::kNumFeatures);
    EXPECT_EQ(std::set<std::string_view>(names.begin(), names.end()).size(), names.size());
    EXPECT_EQ(names.front(), "is_new");
    EXPECT_EQ(tsa::feature_index("large_cancels_1s") < tsa::kNumFeatures, true);
    EXPECT_THROW(tsa::feature_index("no_such_feature"), std::out_of_range);
}

TEST(Features, DescribeTheEventAndItsTraderWindows) {
    tsa::EngineState state(1000);
    std::vector<tsa::Event> events;
    for (std::uint32_t i = 0; i < 3; ++i) {
        events.push_back(make_event(kT0 + i * kMs, i + 1, 7, 1, 10.0, 2000, EventType::New, Side::Buy));
    }
    for (std::uint32_t i = 0; i < 3; ++i) {
        events.push_back(make_event(kT0 + 10 * kMs + i * kMs, i + 1, 7, 1, 10.0, 2000,
                                    EventType::Cancel, Side::Buy));
    }
    for (const auto& e : events) {
        state.update(e);
    }

    const auto row = tsa::compute_features(events.back(), state);

    EXPECT_EQ(get(row, "is_new"), 0.0f);
    EXPECT_EQ(get(row, "is_cancel"), 1.0f);
    EXPECT_EQ(get(row, "is_trade"), 0.0f);
    EXPECT_EQ(get(row, "is_buy"), 1.0f);
    EXPECT_EQ(get(row, "quantity"), 2000.0f);
    EXPECT_EQ(get(row, "orders_1s"), 3.0f);
    EXPECT_EQ(get(row, "cancels_1s"), 3.0f);
    EXPECT_EQ(get(row, "large_cancels_1s"), 3.0f);
    EXPECT_EQ(get(row, "large_cancel_ratio_1s"), 1.0f);
    EXPECT_EQ(get(row, "orders_60s"), 3.0f);
    EXPECT_EQ(get(row, "trades_60s"), 0.0f);
    EXPECT_EQ(get(row, "net_ratio_60s"), 0.0f);
    EXPECT_EQ(get(row, "price_change_pct_60s"), 0.0f);
}

TEST(Features, NetRatioAndPriceChangeArePercentagesOfTheWindow) {
    tsa::EngineState state;
    const std::vector<tsa::Event> events = {
        make_event(kT0, 1, 5, 1, 10.0, 300, EventType::Trade, Side::Buy),
        make_event(kT0 + kMs, 2, 5, 1, 10.5, 100, EventType::Trade, Side::Sell),
    };
    for (const auto& e : events) {
        state.update(e);
    }

    const auto row = tsa::compute_features(events.back(), state);

    EXPECT_FLOAT_EQ(get(row, "net_ratio_60s"), 0.5f);          // (300 - 100) / 400
    EXPECT_FLOAT_EQ(get(row, "price_change_pct_60s"), 5.0f);   // 10.0 -> 10.5
    EXPECT_EQ(get(row, "trades_60s"), 2.0f);
}

TEST(FeatureWriter, WritesFloat32RowsAndADescriptionFile) {
    const auto dir = tsa_test::temp_path("features_out");
    std::filesystem::remove_all(dir);
    const auto path = dir / "x.features.bin";
    tsa::FeatureRow row{};
    row[0] = 1.5f;
    row[tsa::kNumFeatures - 1] = -2.0f;
    {
        tsa::FeatureWriter writer(path);
        writer.write(row);
        writer.write(row);
    }

    EXPECT_EQ(std::filesystem::file_size(path), 2 * tsa::kNumFeatures * sizeof(float));
    std::ifstream in(path, std::ios::binary);
    std::vector<float> back(2 * tsa::kNumFeatures);
    in.read(reinterpret_cast<char*>(back.data()), static_cast<std::streamsize>(back.size() * 4));
    EXPECT_EQ(back[tsa::kNumFeatures], 1.5f);
    EXPECT_EQ(back.back(), -2.0f);

    std::ifstream meta(dir / "x.features.json");
    std::stringstream text;
    text << meta.rdbuf();
    EXPECT_NE(text.str().find("\"rows\":2"), std::string::npos) << text.str();
    EXPECT_NE(text.str().find("\"is_new\""), std::string::npos);
    EXPECT_NE(text.str().find("\"dtype\":\"float32\""), std::string::npos);
}
