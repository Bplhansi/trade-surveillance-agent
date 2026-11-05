// Tests for latency percentiles and the clock measurements.
#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "tsa/latency.hpp"

namespace {

std::vector<std::int64_t> one_to(std::int64_t n) {
    std::vector<std::int64_t> v;
    for (std::int64_t i = 1; i <= n; ++i) {
        v.push_back(i);
    }
    return v;
}

}  // namespace

TEST(Percentile, UsesNearestRankOnOneToHundred) {
    const auto samples = one_to(100);

    EXPECT_EQ(tsa::percentile(samples, 50), 50);
    EXPECT_EQ(tsa::percentile(samples, 95), 95);
    EXPECT_EQ(tsa::percentile(samples, 99), 99);
    EXPECT_EQ(tsa::percentile(samples, 100), 100);
}

TEST(Percentile, DoesNotNeedSortedInput) {
    EXPECT_EQ(tsa::percentile({30, 10, 20}, 50), 20);
}

TEST(Percentile, SingleSampleIsEveryPercentile) {
    EXPECT_EQ(tsa::percentile({7}, 50), 7);
    EXPECT_EQ(tsa::percentile({7}, 99), 7);
}

TEST(Percentile, RejectsEmptyInputAndBadPercentiles) {
    EXPECT_THROW(tsa::percentile({}, 50), std::invalid_argument);
    EXPECT_THROW(tsa::percentile({1, 2}, 0), std::invalid_argument);
    EXPECT_THROW(tsa::percentile({1, 2}, 101), std::invalid_argument);
}

TEST(LatencyRecorder, SummarisesRecordedSamples) {
    tsa::LatencyRecorder recorder(100);
    for (std::int64_t ns = 100; ns >= 1; --ns) {
        recorder.record(ns);
    }

    const auto s = recorder.summarize();
    EXPECT_EQ(s.count, 100u);
    EXPECT_EQ(s.p50_ns, 50);
    EXPECT_EQ(s.p95_ns, 95);
    EXPECT_EQ(s.p99_ns, 99);
    EXPECT_EQ(s.max_ns, 100);
    EXPECT_DOUBLE_EQ(s.mean_ns, 50.5);
}

TEST(LatencyRecorder, SummaryOfNoSamplesIsAllZero) {
    const auto s = tsa::LatencyRecorder(0).summarize();

    EXPECT_EQ(s.count, 0u);
    EXPECT_EQ(s.p99_ns, 0);
    EXPECT_EQ(s.mean_ns, 0.0);
}

TEST(ClockMeasurement, TickIsPositiveAndOverheadIsNotNegative) {
    EXPECT_GT(tsa::measure_clock_tick_ns(), 0);
    EXPECT_GE(tsa::measure_timer_overhead_ns(), 0.0);
}
