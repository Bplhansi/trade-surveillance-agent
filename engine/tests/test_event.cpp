// Tests for the binary event format and the event file reader.
#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <stdexcept>
#include <vector>

#include "test_helpers.hpp"
#include "tsa/event.hpp"

namespace {

// The same 10 events that sim/event_format.py:sample_events() produces.
// Written out independently here so the test really checks Python against C++.
std::vector<tsa::Event> expected_sample_events() {
    std::vector<tsa::Event> events;
    for (std::uint32_t i = 0; i < 10; ++i) {
        events.push_back(tsa::Event{
            .timestamp_ns = 1'700'000'000'000'000'000ULL + i * 1'000ULL,
            .order_id = 1000 + i,
            .trader_id = 1 + i % 3,
            .symbol_id = 1,
            .price = 100.0 + 0.25 * i,
            .quantity = 100 * (i + 1),
            .type = static_cast<tsa::EventType>(i % 3),
            .side = static_cast<tsa::Side>(i % 2),
        });
    }
    return events;
}

std::filesystem::path temp_file(const char* name) {
    return std::filesystem::temp_directory_path() / name;
}

}  // namespace

TEST(EventLayout, MatchesThePythonStructFormat) {
    EXPECT_EQ(sizeof(tsa::Event), 40u);
    EXPECT_EQ(offsetof(tsa::Event, timestamp_ns), 0u);
    EXPECT_EQ(offsetof(tsa::Event, order_id), 8u);
    EXPECT_EQ(offsetof(tsa::Event, trader_id), 12u);
    EXPECT_EQ(offsetof(tsa::Event, symbol_id), 16u);
    EXPECT_EQ(offsetof(tsa::Event, price), 24u);
    EXPECT_EQ(offsetof(tsa::Event, quantity), 32u);
    EXPECT_EQ(offsetof(tsa::Event, type), 36u);
    EXPECT_EQ(offsetof(tsa::Event, side), 37u);
}

TEST(EventValidation, AcceptsTheSampleEvents) {
    for (const auto& e : expected_sample_events()) {
        EXPECT_TRUE(tsa::is_valid(e));
    }
}

TEST(EventValidation, RejectsUnknownTypeSideOrSymbol) {
    auto good = expected_sample_events().front();

    auto bad_type = good;
    bad_type.type = static_cast<tsa::EventType>(7);
    auto bad_side = good;
    bad_side.side = static_cast<tsa::Side>(2);
    auto no_symbol = good;
    no_symbol.symbol_id = 0;
    auto huge_symbol = good;
    huge_symbol.symbol_id = 0xFFFFFFFF;

    EXPECT_FALSE(tsa::is_valid(bad_type));
    EXPECT_FALSE(tsa::is_valid(bad_side));
    EXPECT_FALSE(tsa::is_valid(no_symbol));
    EXPECT_FALSE(tsa::is_valid(huge_symbol));
    EXPECT_TRUE(tsa::is_valid(good));
}

TEST(ReadEvents, ReadsTheTenEventsWrittenByPython) {
    // TSA_PY_EVENTS_FILE is written by sim/write_sample_events.py just before
    // this test runs (see the CTest fixture in engine/CMakeLists.txt).
    const auto events = tsa::read_events(TSA_PY_EVENTS_FILE);
    const auto expected = expected_sample_events();

    ASSERT_EQ(events.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(events[i], expected[i]) << "event #" << i << " differs";
    }
}

TEST(ReadEvents, EmptyFileGivesNoEvents) {
    const auto path = temp_file("tsa_empty_events.bin");
    { std::ofstream out(path, std::ios::binary); }

    EXPECT_TRUE(tsa::read_events(path).empty());
    std::filesystem::remove(path);
}

TEST(ReadEvents, ThrowsWhenFileIsMissing) {
    EXPECT_THROW(tsa::read_events(temp_file("tsa_does_not_exist.bin")), std::runtime_error);
}

TEST(ReadEvents, ThrowsWhenFileHasAPartialEvent) {
    const auto path = temp_file("tsa_partial_event.bin");
    {
        std::ofstream out(path, std::ios::binary);
        const std::vector<char> bytes(41, '\0');  // one full event + 1 stray byte
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    EXPECT_THROW(tsa::read_events(path), std::runtime_error);
    std::filesystem::remove(path);
}
