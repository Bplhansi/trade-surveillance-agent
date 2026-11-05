// Tests for the replay loop and timestamp pacing.
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "test_helpers.hpp"
#include "tsa/event_reader.hpp"
#include "tsa/latency.hpp"
#include "tsa/market_state.hpp"
#include "tsa/replay.hpp"

using namespace std::chrono_literals;
using tsa_test::make_sequence;

namespace {

constexpr std::uint64_t kSecond = 1'000'000'000;

// A state that just remembers what it saw.
struct RecordingState {
    std::vector<tsa::Event> seen;
    void update(const tsa::Event& e) { seen.push_back(e); }
};

// A rule that raises an alert for every Cancel.
struct AlertOnCancel {
    template <class State>
    std::optional<int> check(const tsa::Event& e, const State&) const {
        return e.type == tsa::EventType::Cancel ? std::optional<int>(1) : std::nullopt;
    }
};

// A rule that remembers what it has seen (non-const check), alerting on every 2nd event.
struct EverySecondEvent {
    int count = 0;
    template <class State>
    std::optional<int> check(const tsa::Event&, const State&) {
        return ++count % 2 == 0 ? std::optional<int>(count) : std::nullopt;
    }
};

auto no_sleep = [](std::chrono::nanoseconds) {};

}  // namespace

// --- Pacer ------------------------------------------------------------------------

TEST(Pacer, MaxSpeedNeverWaits) {
    tsa::Pacer pacer;  // default = as fast as possible

    EXPECT_FALSE(pacer.paced());
    EXPECT_EQ(pacer.delay(0, 0ns), 0ns);
    EXPECT_EQ(pacer.delay(10 * kSecond, 0ns), 0ns);
}

TEST(Pacer, WaitsHalfTheGapAtDoubleSpeed) {
    tsa::Pacer pacer(2.0);
    const std::uint64_t t0 = 5 * kSecond;

    EXPECT_TRUE(pacer.paced());
    EXPECT_EQ(pacer.delay(t0, 0ns), 0ns);                // first event: start now
    EXPECT_EQ(pacer.delay(t0 + kSecond, 0ns), 500ms);    // 1 s of market time at 2x
    EXPECT_EQ(pacer.delay(t0 + kSecond, 200ms), 300ms);  // some time already passed
    EXPECT_EQ(pacer.delay(t0 + kSecond, 900ms), 0ns);    // running late: don't wait
}

TEST(Pacer, RejectsZeroOrNegativeSpeed) {
    EXPECT_THROW(tsa::Pacer(0.0), std::invalid_argument);
    EXPECT_THROW(tsa::Pacer(-1.0), std::invalid_argument);
}

// --- Replay loop ------------------------------------------------------------------

TEST(Replay, ProcessesEveryEventInOrderAndTimesEachOne) {
    const auto events = make_sequence(10);
    RecordingState state;
    tsa::Pacer pacer;
    tsa::LatencyRecorder recorder(events.size());

    const auto result = tsa::replay(events, state, tsa::NoRules{}, pacer, recorder, no_sleep);

    EXPECT_EQ(state.seen, events);
    EXPECT_EQ(result.events, 10u);
    EXPECT_EQ(result.alerts, 0u);
    EXPECT_EQ(result.latency.count, 10u);
    EXPECT_GE(result.wall_ns, 0);
}

TEST(Replay, CountsAlertsFromTheRules) {
    const auto events = make_sequence(9);  // types cycle New, Cancel, Trade -> 3 cancels
    RecordingState state;
    tsa::Pacer pacer;
    tsa::LatencyRecorder recorder(events.size());

    const auto result = tsa::replay(events, state, AlertOnCancel{}, pacer, recorder, no_sleep);

    EXPECT_EQ(result.alerts, 3u);
}

TEST(Replay, SupportsRulesWithMemoryAndHandsEachAlertOver) {
    const auto events = make_sequence(6);
    RecordingState state;
    tsa::Pacer pacer;
    tsa::LatencyRecorder recorder(events.size());
    EverySecondEvent rules;
    std::vector<int> handed_over;

    const auto result = tsa::replay(events, state, rules, pacer, recorder, no_sleep,
                                    [&](const int& alert) { handed_over.push_back(alert); });

    EXPECT_EQ(rules.count, 6);  // the caller's rules object was used, not a copy
    EXPECT_EQ(result.alerts, 3u);
    EXPECT_EQ(handed_over, (std::vector<int>{2, 4, 6}));
}

TEST(Replay, PacedReplaySleepsAccordingToTheTimestampGaps) {
    std::vector<tsa::Event> events = make_sequence(4);
    for (std::size_t i = 0; i < events.size(); ++i) {
        events[i].timestamp_ns = kSecond * (i + 1);  // one event per second
    }
    std::vector<std::chrono::nanoseconds> sleeps;
    RecordingState state;
    tsa::Pacer pacer(2.0);
    tsa::LatencyRecorder recorder(events.size());

    tsa::replay(events, state, tsa::NoRules{}, pacer, recorder,
                [&](std::chrono::nanoseconds d) { sleeps.push_back(d); });

    // The fake sleep doesn't advance time, so the k-th wait is about k * 0.5 s.
    ASSERT_EQ(sleeps.size(), 3u);
    for (std::size_t k = 0; k < sleeps.size(); ++k) {
        const auto expected = std::chrono::nanoseconds(500ms) * static_cast<int>(k + 1);
        EXPECT_NEAR(static_cast<double>(sleeps[k].count()), static_cast<double>(expected.count()),
                    20e6)
            << "wait #" << k;
    }
}

TEST(Replay, RunsTheRealReaderAndStateOnThePythonFile) {
    tsa::EventReader reader(TSA_PY_EVENTS_FILE);
    tsa::MarketState state;
    tsa::Pacer pacer;
    tsa::LatencyRecorder recorder(reader.total_events());

    const auto result = tsa::replay(reader, state, tsa::NoRules{}, pacer, recorder, no_sleep);

    EXPECT_EQ(result.events, 10u);
    EXPECT_EQ(state.count(tsa::EventType::New) + state.count(tsa::EventType::Cancel) +
                  state.count(tsa::EventType::Trade),
              10u);
}
