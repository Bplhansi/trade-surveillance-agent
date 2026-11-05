#pragma once

// Records per-event processing times and turns them into percentiles.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tsa {

struct LatencySummary {
    std::size_t count = 0;
    double mean_ns = 0.0;
    std::int64_t p50_ns = 0;
    std::int64_t p95_ns = 0;
    std::int64_t p99_ns = 0;
    std::int64_t max_ns = 0;
};

// Nearest-rank percentile: the smallest sample with at least p% of samples at or below it.
// p must be in (0, 100]. Throws std::invalid_argument for empty input or a bad p.
std::int64_t percentile(std::vector<std::int64_t> samples, double p);

class LatencyRecorder {
public:
    // Reserves space up front so record() never allocates in the hot loop.
    explicit LatencyRecorder(std::size_t expected_samples) { samples_.reserve(expected_samples); }

    void record(std::int64_t ns) { samples_.push_back(ns); }
    std::size_t count() const noexcept { return samples_.size(); }

    // All zero when nothing was recorded.
    LatencySummary summarize() const;

private:
    std::vector<std::int64_t> samples_;
};

// Smallest step std::chrono::steady_clock can show on this machine, in nanoseconds.
std::int64_t measure_clock_tick_ns();

// Average cost of one steady_clock::now() call, in nanoseconds.
double measure_timer_overhead_ns();

}  // namespace tsa
