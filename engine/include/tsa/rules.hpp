#pragma once

// Baseline rules, one per pattern. Thresholds come from
// config/rules.ini; see that file for what each one means.

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <unordered_map>

#include "tsa/config.hpp"
#include "tsa/engine_state.hpp"
#include "tsa/event.hpp"
#include "tsa/rolling_stats.hpp"

namespace tsa {

enum class Rule : std::uint8_t { Spoofing = 0, WashTrade = 1, PumpAndDump = 2 };

// Matches the pattern names in the Python labels files.
constexpr std::string_view rule_name(Rule rule) noexcept {
    switch (rule) {
        case Rule::Spoofing: return "spoofing";
        case Rule::WashTrade: return "wash_trade";
        case Rule::PumpAndDump: return "pump_and_dump";
    }
    return "unknown";
}

struct SpoofingConfig {
    std::uint32_t large_order_qty = kDefaultLargeOrderQty;
    std::uint32_t min_large_cancels = 3;
    double min_large_cancel_ratio = 0.8;
};

struct WashTradeConfig {
    std::uint32_t min_trades = 6;
    std::uint64_t min_volume = 500;
    double max_net_ratio = 0.05;
    std::uint32_t min_self_trades = 0;  // 0 = the guide's plain balance rule
};

struct PumpAndDumpConfig {
    std::int64_t min_net_qty = 2000;
    double min_rise_pct = 1.0;
    std::uint64_t min_sell_qty = 1000;
    double sell_within_s = 600;
};

// The ONNX model: which events it scores, and the score that raises an alert.
struct ModelConfig {
    double threshold = std::numeric_limits<double>::infinity();  // off unless configured
    std::uint32_t gate_min_large_cancels_1s = 1;  // score a Cancel only after this many
    std::uint32_t gate_min_trades_60s = 2;        // score a Trade only after this many
};

struct RulesConfig {
    double cooldown_s = 60;
    SpoofingConfig spoofing;
    WashTradeConfig wash_trade;
    PumpAndDumpConfig pump_and_dump;
    ModelConfig model;

    // Reads [general], [spoofing], [wash_trade], [pump_and_dump] and [model]. Missing keys
    // keep their defaults; unknown keys (typos) and out-of-range values throw std::runtime_error.
    static RulesConfig from(const Config& config);
};

enum class AlertSource : std::uint8_t { Rules = 0, Model = 1 };

struct Alert {
    Rule rule = Rule::Spoofing;
    Window window = Window::OneSecond;  // which window the evidence comes from
    std::uint64_t timestamp_ns = 0;
    std::uint64_t event_index = 0;      // row of the triggering event in the events file
    std::uint32_t trader_id = 0;
    std::uint32_t symbol_id = 0;
    WindowStats evidence;               // the statistics the rule looked at
    AlertSource source = AlertSource::Rules;
    std::optional<double> model_score;  // -log10 P(normal) from the model, if one is loaded
};

class Rules {
public:
    explicit Rules(RulesConfig config = {});

    // Call once per event, in file order, after the state has been updated with it.
    std::optional<Alert> check(const Event& e, const EngineState& state);

    const RulesConfig& config() const noexcept { return config_; }

private:
    std::optional<Alert> check_spoofing(const Event& e, const EngineState& state);
    std::optional<Alert> check_pump_and_dump(const Event& e, const EngineState& state);
    std::optional<Alert> check_wash_trade(const Event& e, const EngineState& state);
    std::optional<Alert> raise(Rule rule, Window window, const Event& e, const WindowStats& stats);

    RulesConfig config_;
    std::uint64_t cooldown_ns_;
    std::uint64_t sell_within_ns_;
    std::uint64_t event_index_ = 0;
    std::array<std::unordered_map<std::uint64_t, std::uint64_t>, 3> last_alert_ns_;  // per rule
    std::unordered_map<std::uint64_t, std::uint64_t> pump_marked_ns_;  // trader+symbol -> time
};

}  // namespace tsa
