#pragma once

// The engine's main loop:
//
//   for (const Event& e : reader) {
//       state.update(e);
//       if (auto alert = rules.check(e, state)) { ... }
//   }
//
// Each event's update + check is timed with std::chrono::steady_clock.

#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

#include "tsa/event.hpp"
#include "tsa/latency.hpp"

namespace tsa {

// Rules that never raise an alert (for measuring the engine without rules).
struct NoRules {
    struct NoAlert {};
    template <class State>
    std::optional<NoAlert> check(const Event&, const State&) const noexcept {
        return std::nullopt;
    }
};

// Decides how long to wait before each event so the replay follows the event timestamps.
// Default: no waiting (as fast as possible). Pacer(10.0) replays at 10x real time.
class Pacer {
public:
    Pacer() = default;
    explicit Pacer(double speed) : speed_(speed) {
        if (!(speed > 0.0)) {
            throw std::invalid_argument("replay speed must be greater than 0");
        }
    }

    bool paced() const noexcept { return speed_ > 0.0; }

    // How long to wait before an event with this timestamp, given how much real time
    // has passed since the replay started. The first event sets the starting point.
    std::chrono::nanoseconds delay(std::uint64_t event_ts_ns, std::chrono::nanoseconds elapsed) {
        if (!paced()) {
            return std::chrono::nanoseconds{0};
        }
        if (!started_) {
            first_ts_ns_ = event_ts_ns;
            started_ = true;
        }
        const auto market_gap = static_cast<double>(event_ts_ns - first_ts_ns_);
        const auto target = std::chrono::nanoseconds(static_cast<std::int64_t>(market_gap / speed_));
        return target > elapsed ? target - elapsed : std::chrono::nanoseconds{0};
    }

private:
    double speed_ = 0.0;  // 0 = as fast as possible
    bool started_ = false;
    std::uint64_t first_ts_ns_ = 0;
};

struct ReplayResult {
    std::uint64_t events = 0;
    std::uint64_t alerts = 0;
    std::int64_t wall_ns = 0;  // whole run, including reading the file
    LatencySummary latency;    // per event: state.update + rules.check
};

// Does nothing with an alert (used when no alerts file is wanted).
struct IgnoreAlert {
    template <class Alert>
    void operator()(const Alert&) const noexcept {}
};

// `rules` may keep memory between events (e.g. who is pumping a price), so it is taken by
// reference and the caller's object is used. Each alert goes to `on_alert` *after* the timer
// stops, so writing alerts to a file is not counted in the per-event latency.
template <class Events, class State, class Rules, class SleepFn, class OnAlert = IgnoreAlert>
ReplayResult replay(Events&& events, State& state, Rules&& rules, Pacer& pacer,
                    LatencyRecorder& recorder, SleepFn&& sleep, OnAlert&& on_alert = {}) {
    using Clock = std::chrono::steady_clock;
    ReplayResult result;
    const auto start = Clock::now();

    for (const Event& e : events) {
        if (pacer.paced()) {
            const auto wait = pacer.delay(e.timestamp_ns, Clock::now() - start);
            if (wait.count() > 0) {
                sleep(wait);
            }
        }

        const auto t0 = Clock::now();
        state.update(e);
        const auto alert = rules.check(e, state);
        const auto t1 = Clock::now();

        recorder.record(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        if (alert) {
            ++result.alerts;
            on_alert(*alert);
        }
        ++result.events;
    }

    result.wall_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
    result.latency = recorder.summarize();
    return result;
}

// Same as above, sleeping for real when paced.
template <class Events, class State, class Rules, class OnAlert = IgnoreAlert>
ReplayResult replay_realtime(Events&& events, State& state, Rules&& rules, Pacer& pacer,
                             LatencyRecorder& recorder, OnAlert&& on_alert = {}) {
    return replay(std::forward<Events>(events), state, std::forward<Rules>(rules), pacer, recorder,
                  [](std::chrono::nanoseconds d) { std::this_thread::sleep_for(d); },
                  std::forward<OnAlert>(on_alert));
}

}  // namespace tsa
