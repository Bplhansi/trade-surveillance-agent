#include "tsa/latency.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace tsa {

namespace {

// Nearest-rank index into a sorted list of n samples.
std::size_t rank_index(std::size_t n, double p) {
    const auto rank = static_cast<std::size_t>(std::ceil(p / 100.0 * static_cast<double>(n)));
    return std::clamp<std::size_t>(rank, 1, n) - 1;
}

}  // namespace

std::int64_t percentile(std::vector<std::int64_t> samples, double p) {
    if (samples.empty()) {
        throw std::invalid_argument("percentile of an empty list");
    }
    if (!(p > 0.0 && p <= 100.0)) {
        throw std::invalid_argument("percentile must be in (0, 100]");
    }
    const auto idx = rank_index(samples.size(), p);
    std::nth_element(samples.begin(), samples.begin() + static_cast<std::ptrdiff_t>(idx),
                     samples.end());
    return samples[idx];
}

LatencySummary LatencyRecorder::summarize() const {
    LatencySummary s;
    s.count = samples_.size();
    if (samples_.empty()) {
        return s;
    }
    std::vector<std::int64_t> sorted = samples_;
    std::sort(sorted.begin(), sorted.end());
    const auto n = sorted.size();
    s.mean_ns = static_cast<double>(std::accumulate(sorted.begin(), sorted.end(), std::int64_t{0})) /
                static_cast<double>(n);
    s.p50_ns = sorted[rank_index(n, 50)];
    s.p95_ns = sorted[rank_index(n, 95)];
    s.p99_ns = sorted[rank_index(n, 99)];
    s.max_ns = sorted.back();
    return s;
}

std::int64_t measure_clock_tick_ns() {
    using Clock = std::chrono::steady_clock;
    std::int64_t smallest = std::numeric_limits<std::int64_t>::max();
    for (int i = 0; i < 1000; ++i) {
        const auto t0 = Clock::now();
        auto t1 = Clock::now();
        while (t1 == t0) {
            t1 = Clock::now();
        }
        smallest = std::min<std::int64_t>(
            smallest, std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }
    return smallest;
}

double measure_timer_overhead_ns() {
    using Clock = std::chrono::steady_clock;
    constexpr int kCalls = 1'000'000;
    volatile std::int64_t sink = 0;  // stops the compiler removing the calls
    const auto start = Clock::now();
    for (int i = 0; i < kCalls; ++i) {
        sink = sink + Clock::now().time_since_epoch().count();
    }
    const auto total = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start);
    return static_cast<double>(total.count()) / kCalls;
}

}  // namespace tsa
