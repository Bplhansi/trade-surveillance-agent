// tsa_replay: replays an event file through the engine and prints throughput and latency.
//
//   tsa_replay <events.bin> [--rules <rules.ini>] [--alerts <alerts.jsonl>]
//              [--speed max|<factor>] [--repeat <n>]
//
// --rules      run the baseline rules with thresholds from this file (default: no rules)
// --alerts     write every alert to this file as JSON Lines (needs --rules; last run only)
// --features   also write one feature row per event (float32) for model training
// --model      also score events with this ONNX model (needs --rules; [model] section)
// --speed max  (default) as fast as possible, for benchmarking
// --speed 10   follow the event timestamps at 10x real time
// --repeat 5   run 5 times and report the median run (and the range)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "tsa/alert_writer.hpp"
#include "tsa/config.hpp"
#include "tsa/detector.hpp"
#include "tsa/engine_state.hpp"
#include "tsa/event_reader.hpp"
#include "tsa/features.hpp"
#include "tsa/latency.hpp"
#include "tsa/model.hpp"
#include "tsa/replay.hpp"
#include "tsa/rules.hpp"
#include "tsa/version.hpp"

namespace {

// Set by CMake: the real build configuration, and the optimisation flag (empty unless Release).
#ifndef TSA_BUILD_CONFIG
#define TSA_BUILD_CONFIG "unknown"
#endif
#ifndef TSA_BUILD_OPT
#define TSA_BUILD_OPT ""
#endif

struct Options {
    std::string path;
    std::string rules_path;   // empty = no rules
    std::string alerts_path;  // empty = don't write alerts
    std::string features_path;  // empty = don't export features
    std::string model_path;     // empty = rules only
    double speed = 0.0;  // 0 = max
    int repeat = 1;
};

[[noreturn]] void usage_error(const std::string& message) {
    std::cerr << "error: " << message << "\n"
              << "usage: tsa_replay <events.bin> [--rules <rules.ini>] [--alerts <alerts.jsonl>]\n"
              << "                  [--model <model.onnx>] [--features <out.bin>]\n"
              << "                  [--speed max|<factor>] [--repeat <n>]\n";
    std::exit(2);
}

// Parses the whole string as a number; nullopt for "abc", "10abc", "2.5" (as int), overflow.
template <class T>
std::optional<T> parse_number(const std::string& text) {
    try {
        std::size_t used = 0;
        T value{};
        if constexpr (std::is_same_v<T, int>) {
            value = std::stoi(text, &used);
        } else {
            value = std::stod(text, &used);
        }
        return used == text.size() ? std::optional<T>(value) : std::nullopt;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

Options parse(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--speed" && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value != "max") {
                const auto speed = parse_number<double>(value);
                if (!speed || !(*speed > 0.0) || !std::isfinite(*speed)) {
                    usage_error("--speed must be a number greater than 0, or 'max'");
                }
                opt.speed = *speed;
            }
        } else if (arg == "--repeat" && i + 1 < argc) {
            const auto repeat = parse_number<int>(argv[++i]);
            if (!repeat || *repeat < 1) usage_error("--repeat must be a whole number, at least 1");
            opt.repeat = *repeat;
        } else if (arg == "--rules" && i + 1 < argc) {
            opt.rules_path = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            opt.model_path = argv[++i];
        } else if (arg == "--features" && i + 1 < argc) {
            opt.features_path = argv[++i];
        } else if (arg == "--alerts" && i + 1 < argc) {
            opt.alerts_path = argv[++i];
        } else if (!arg.empty() && arg[0] != '-' && opt.path.empty()) {
            opt.path = arg;
        } else {
            usage_error("unexpected argument '" + arg + "'");
        }
    }
    if (opt.path.empty()) usage_error("missing event file");
    if (!opt.alerts_path.empty() && opt.rules_path.empty()) usage_error("--alerts needs --rules");
    if (!opt.model_path.empty() && opt.rules_path.empty()) usage_error("--model needs --rules");
    return opt;
}

// 1234567 -> "1,234,567"
std::string with_commas(std::uint64_t n) {
    std::string digits = std::to_string(n);
    for (int i = static_cast<int>(digits.size()) - 3; i > 0; i -= 3) {
        digits.insert(static_cast<std::size_t>(i), ",");
    }
    return digits;
}

template <class T>
T median_of(std::vector<T> values) {
    std::sort(values.begin(), values.end());
    return values[(values.size() - 1) / 2];
}

template <class T, class F>
std::vector<T> collect(const std::vector<tsa::ReplayResult>& runs, F field) {
    std::vector<T> out;
    for (const auto& r : runs) out.push_back(field(r));
    return out;
}

double events_per_second(const tsa::ReplayResult& r) {
    return r.wall_ns > 0 ? static_cast<double>(r.events) * 1e9 / static_cast<double>(r.wall_ns)
                         : 0.0;
}

void print_metric(const char* label, std::vector<std::int64_t> values, const char* unit) {
    const auto [lo, hi] = std::minmax_element(values.begin(), values.end());
    std::printf("%-18s %10s %s", label, with_commas(static_cast<std::uint64_t>(median_of(values))).c_str(), unit);
    if (values.size() > 1) {
        std::printf("   (range %s - %s)", with_commas(static_cast<std::uint64_t>(*lo)).c_str(),
                    with_commas(static_cast<std::uint64_t>(*hi)).c_str());
    }
    std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
    const Options opt = parse(argc, argv);
    try {
        const auto tick = tsa::measure_clock_tick_ns();
        const auto overhead = tsa::measure_timer_overhead_ns();
        std::printf("tsa_replay %s  |  file: %s\n", std::string(tsa::version()).c_str(), opt.path.c_str());
        std::printf("build: %s %s  |  speed: %s  |  runs: %d\n", TSA_BUILD_CONFIG, TSA_BUILD_OPT,
                    opt.speed > 0 ? (std::to_string(opt.speed) + "x").c_str() : "max", opt.repeat);
        std::printf("clock tick: %lld ns  |  timer overhead: %.1f ns per now() call\n\n",
                    static_cast<long long>(tick), overhead);

        std::optional<tsa::RulesConfig> rules_config;
        if (!opt.rules_path.empty()) {
            rules_config = tsa::RulesConfig::from(tsa::Config::from_file(opt.rules_path));
            std::printf("rules: %s\n", opt.rules_path.c_str());
        } else {
            std::printf("rules: none (engine only)\n");
        }
        std::unique_ptr<tsa::ModelScorer> model;
        if (!opt.model_path.empty()) {
            model = std::make_unique<tsa::ModelScorer>(opt.model_path);
            std::printf("model: %s  |  threshold %g  |  gate: %u large cancel(s) in 1 s, "
                        "%u trade(s) in 60 s\n", opt.model_path.c_str(),
                        rules_config->model.threshold,
                        rules_config->model.gate_min_large_cancels_1s,
                        rules_config->model.gate_min_trades_60s);
        }
        std::uint64_t model_scored = 0;

        if (!opt.features_path.empty()) {
            // Separate, untimed pass so exporting doesn't distort the speed numbers.
            tsa::EventReader reader(opt.path);
            tsa::EngineState state(rules_config ? rules_config->spoofing.large_order_qty
                                                : tsa::kDefaultLargeOrderQty);
            tsa::FeatureWriter writer(opt.features_path);
            for (const tsa::Event& e : reader) {
                state.update(e);
                writer.write(tsa::compute_features(e, state));
            }
            std::printf("features written to: %s (%s rows)\n", opt.features_path.c_str(),
                        with_commas(writer.rows()).c_str());
        }

        std::vector<tsa::ReplayResult> runs;
        for (int i = 0; i < opt.repeat; ++i) {
            tsa::EventReader reader(opt.path);
            tsa::Pacer pacer = opt.speed > 0 ? tsa::Pacer(opt.speed) : tsa::Pacer();
            tsa::LatencyRecorder recorder(reader.total_events());
            tsa::ReplayResult r;
            if (rules_config) {
                tsa::EngineState state(rules_config->spoofing.large_order_qty);
                tsa::Detector detector(*rules_config, model.get());
                // Every run raises the same alerts, so only the last run writes them.
                std::optional<tsa::AlertWriter> writer;
                if (!opt.alerts_path.empty() && i + 1 == opt.repeat) {
                    writer.emplace(opt.alerts_path);
                }
                r = tsa::replay_realtime(reader, state, detector, pacer, recorder,
                                         [&](const tsa::Alert& a) {
                                             if (writer) writer->write(a);
                                         });
                model_scored = detector.scored();
            } else {
                tsa::EngineState state;
                r = tsa::replay_realtime(reader, state, tsa::NoRules{}, pacer, recorder);
            }
            runs.push_back(r);
            std::printf("run %d: %s events in %.3f s  ->  %s events/s  p50 %lld ns  p99 %lld ns\n",
                        i + 1, with_commas(r.events).c_str(), static_cast<double>(r.wall_ns) / 1e9,
                        with_commas(static_cast<std::uint64_t>(events_per_second(r))).c_str(),
                        static_cast<long long>(r.latency.p50_ns),
                        static_cast<long long>(r.latency.p99_ns));
        }

        if (!opt.alerts_path.empty()) {
            std::printf("alerts written to: %s\n", opt.alerts_path.c_str());
        }
        if (model) {
            std::printf("model scored %s events (last run)\n",
                        with_commas(model_scored).c_str());
        }

        std::printf("\n%s%s\n", "summary", opt.repeat > 1 ? " (median of runs)" : "");
        std::printf("%-18s %10s\n", "events", with_commas(runs.front().events).c_str());
        print_metric("throughput", collect<std::int64_t>(runs, [](const auto& r) {
            return static_cast<std::int64_t>(events_per_second(r)); }), "events/s");
        print_metric("latency mean", collect<std::int64_t>(runs, [](const auto& r) {
            return static_cast<std::int64_t>(r.latency.mean_ns + 0.5); }), "ns");
        print_metric("latency p50", collect<std::int64_t>(runs, [](const auto& r) {
            return r.latency.p50_ns; }), "ns");
        print_metric("latency p95", collect<std::int64_t>(runs, [](const auto& r) {
            return r.latency.p95_ns; }), "ns");
        print_metric("latency p99", collect<std::int64_t>(runs, [](const auto& r) {
            return r.latency.p99_ns; }), "ns");
        print_metric("latency max", collect<std::int64_t>(runs, [](const auto& r) {
            return r.latency.max_ns; }), "ns");
        print_metric("alerts", collect<std::int64_t>(runs, [](const auto& r) {
            return static_cast<std::int64_t>(r.alerts); }), "");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
