#include "tsa/rules.hpp"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace tsa {

namespace {

constexpr double kNsPerSecond = 1e9;

std::uint64_t pair_key(const Event& e) noexcept {
    return (static_cast<std::uint64_t>(e.trader_id) << 32) | e.symbol_id;
}

const std::set<std::string, std::less<>> kKnownKeys = {
    "general.cooldown_s",
    "spoofing.large_order_qty", "spoofing.min_large_cancels", "spoofing.min_large_cancel_ratio",
    "wash_trade.min_trades", "wash_trade.min_volume", "wash_trade.max_net_ratio",
    "wash_trade.min_self_trades",
    "pump_and_dump.min_net_qty", "pump_and_dump.min_rise_pct", "pump_and_dump.min_sell_qty",
    "pump_and_dump.sell_within_s",
    "model.threshold", "model.gate_min_large_cancels_1s", "model.gate_min_trades_60s",
};

[[noreturn]] void bad_value(const Config& c, std::string_view section, std::string_view key,
                            const std::string& rule) {
    throw std::runtime_error(c.source() + ": " + std::string(section) + "." + std::string(key) +
                             " must be " + rule);
}

// A whole number between 0 and `max`.
template <class T>
T read_count(const Config& c, std::string_view section, std::string_view key, T fallback) {
    const double v = c.get_double(section, key, static_cast<double>(fallback));
    if (!(v >= 0) || v != std::floor(v) || v > static_cast<double>(std::numeric_limits<T>::max())) {
        bad_value(c, section, key, "a whole number, 0 or more");
    }
    return static_cast<T>(v);
}

double read_ratio(const Config& c, std::string_view section, std::string_view key, double fallback) {
    const double v = c.get_double(section, key, fallback);
    if (!(v >= 0.0 && v <= 1.0)) {
        bad_value(c, section, key, "between 0 and 1");
    }
    return v;
}

double read_positive(const Config& c, std::string_view section, std::string_view key,
                     double fallback) {
    const double v = c.get_double(section, key, fallback);
    if (!(v > 0.0) || !std::isfinite(v)) {
        bad_value(c, section, key, "a number greater than 0");
    }
    return v;
}

}  // namespace

RulesConfig RulesConfig::from(const Config& c) {
    for (const auto& key : c.keys()) {
        if (!kKnownKeys.contains(key)) {
            throw std::runtime_error(c.source() + ": unknown setting '" + key +
                                     "' (misspelled?)");
        }
    }
    const RulesConfig d;  // defaults
    RulesConfig r;
    r.cooldown_s = read_positive(c, "general", "cooldown_s", d.cooldown_s);
    r.spoofing.large_order_qty =
        read_count<std::uint32_t>(c, "spoofing", "large_order_qty", d.spoofing.large_order_qty);
    r.spoofing.min_large_cancels =
        read_count<std::uint32_t>(c, "spoofing", "min_large_cancels", d.spoofing.min_large_cancels);
    r.spoofing.min_large_cancel_ratio =
        read_ratio(c, "spoofing", "min_large_cancel_ratio", d.spoofing.min_large_cancel_ratio);
    r.wash_trade.min_trades =
        read_count<std::uint32_t>(c, "wash_trade", "min_trades", d.wash_trade.min_trades);
    r.wash_trade.min_volume =
        read_count<std::uint64_t>(c, "wash_trade", "min_volume", d.wash_trade.min_volume);
    r.wash_trade.max_net_ratio =
        read_ratio(c, "wash_trade", "max_net_ratio", d.wash_trade.max_net_ratio);
    r.wash_trade.min_self_trades =
        read_count<std::uint32_t>(c, "wash_trade", "min_self_trades", d.wash_trade.min_self_trades);
    r.pump_and_dump.min_net_qty =
        read_count<std::int64_t>(c, "pump_and_dump", "min_net_qty", d.pump_and_dump.min_net_qty);
    r.pump_and_dump.min_rise_pct =
        c.get_double("pump_and_dump", "min_rise_pct", d.pump_and_dump.min_rise_pct);
    if (!(r.pump_and_dump.min_rise_pct >= 0.0)) {
        bad_value(c, "pump_and_dump", "min_rise_pct", "0 or more");
    }
    r.pump_and_dump.min_sell_qty =
        read_count<std::uint64_t>(c, "pump_and_dump", "min_sell_qty", d.pump_and_dump.min_sell_qty);
    r.pump_and_dump.sell_within_s =
        read_positive(c, "pump_and_dump", "sell_within_s", d.pump_and_dump.sell_within_s);
    r.model.threshold = c.get_double("model", "threshold", d.model.threshold);
    if (std::isnan(r.model.threshold)) {
        bad_value(c, "model", "threshold", "a number");
    }
    r.model.gate_min_large_cancels_1s = read_count<std::uint32_t>(
        c, "model", "gate_min_large_cancels_1s", d.model.gate_min_large_cancels_1s);
    r.model.gate_min_trades_60s =
        read_count<std::uint32_t>(c, "model", "gate_min_trades_60s", d.model.gate_min_trades_60s);
    return r;
}

Rules::Rules(RulesConfig config)
    : config_(config),
      cooldown_ns_(static_cast<std::uint64_t>(config.cooldown_s * kNsPerSecond)),
      sell_within_ns_(static_cast<std::uint64_t>(config.pump_and_dump.sell_within_s * kNsPerSecond)) {}

std::optional<Alert> Rules::check(const Event& e, const EngineState& state) {
    const auto index = event_index_++;
    std::optional<Alert> alert;
    if (!is_valid(e)) {
        return alert;
    }
    if (e.type == EventType::Cancel) {
        alert = check_spoofing(e, state);
    } else if (e.type == EventType::Trade) {
        alert = check_pump_and_dump(e, state);
        if (!alert) {
            alert = check_wash_trade(e, state);
        }
    }
    if (alert) {
        alert->event_index = index;
    }
    return alert;
}

std::optional<Alert> Rules::check_spoofing(const Event& e, const EngineState& state) {
    const auto& cfg = config_.spoofing;
    if (e.quantity < cfg.large_order_qty) {
        return std::nullopt;
    }
    const auto s = state.rolling.stats(e.trader_id, e.symbol_id, Window::OneSecond, e.timestamp_ns);
    if (s.large_cancels >= cfg.min_large_cancels &&
        s.large_cancel_ratio >= cfg.min_large_cancel_ratio) {
        return raise(Rule::Spoofing, Window::OneSecond, e, s);
    }
    return std::nullopt;
}

std::optional<Alert> Rules::check_pump_and_dump(const Event& e, const EngineState& state) {
    const auto& cfg = config_.pump_and_dump;
    const auto s =
        state.rolling.stats(e.trader_id, e.symbol_id, Window::SixtySeconds, e.timestamp_ns);

    if (e.side == Side::Buy) {
        const bool rose = s.first_trade_price > 0.0 &&
                          s.price_change / s.first_trade_price * 100.0 >= cfg.min_rise_pct;
        if (s.net_qty >= cfg.min_net_qty && rose) {
            pump_marked_ns_[pair_key(e)] = e.timestamp_ns;  // this account is pushing the price up
        }
        return std::nullopt;
    }

    const auto marked = pump_marked_ns_.find(pair_key(e));
    if (marked != pump_marked_ns_.end() && e.timestamp_ns - marked->second <= sell_within_ns_ &&
        s.sell_qty >= cfg.min_sell_qty) {
        return raise(Rule::PumpAndDump, Window::SixtySeconds, e, s);
    }
    return std::nullopt;
}

std::optional<Alert> Rules::check_wash_trade(const Event& e, const EngineState& state) {
    const auto& cfg = config_.wash_trade;
    const auto s =
        state.rolling.stats(e.trader_id, e.symbol_id, Window::SixtySeconds, e.timestamp_ns);
    const auto volume = s.buy_qty + s.sell_qty;
    const auto net = static_cast<double>(std::llabs(s.net_qty));
    if (s.trades >= cfg.min_trades && s.self_trades >= cfg.min_self_trades &&
        s.buy_qty >= cfg.min_volume &&
        s.sell_qty >= cfg.min_volume && net <= cfg.max_net_ratio * static_cast<double>(volume)) {
        return raise(Rule::WashTrade, Window::SixtySeconds, e, s);
    }
    return std::nullopt;
}

std::optional<Alert> Rules::raise(Rule rule, Window window, const Event& e,
                                  const WindowStats& stats) {
    auto& last = last_alert_ns_[static_cast<std::size_t>(rule)];
    const auto key = pair_key(e);
    const auto it = last.find(key);
    if (it != last.end() && e.timestamp_ns - it->second < cooldown_ns_) {
        return std::nullopt;  // already alerted on this trader + symbol recently
    }
    last[key] = e.timestamp_ns;
    return Alert{.rule = rule, .window = window, .timestamp_ns = e.timestamp_ns, .event_index = 0,
                 .trader_id = e.trader_id, .symbol_id = e.symbol_id, .evidence = stats};
}

}  // namespace tsa
