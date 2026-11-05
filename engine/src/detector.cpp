#include "tsa/detector.hpp"

namespace tsa {

namespace {

const std::size_t kLargeCancels1s = feature_index("large_cancels_1s");
const std::size_t kTrades60s = feature_index("trades_60s");

}  // namespace

bool passes_gate(const Event& e, const FeatureRow& row, const ModelConfig& config) noexcept {
    switch (e.type) {
        case EventType::Cancel:
            return row[kLargeCancels1s] >= static_cast<float>(config.gate_min_large_cancels_1s);
        case EventType::Trade:
            return row[kTrades60s] >= static_cast<float>(config.gate_min_trades_60s);
        case EventType::New:
            break;
    }
    return false;
}

ModelCooldownKey model_cooldown_key(Rule pattern, std::uint32_t trader_id,
                                    std::uint32_t symbol_id) noexcept {
    return {.pattern = pattern, .trader_id = trader_id, .symbol_id = symbol_id};
}

Detector::Detector(RulesConfig config, ModelScorer* model)
    : rules_(config),
      model_(model),
      model_config_(config.model),
      cooldown_ns_(static_cast<std::uint64_t>(config.cooldown_s * 1e9)) {}

std::optional<Alert> Detector::check(const Event& e, const EngineState& state) {
    const auto index = event_index_++;
    auto alert = rules_.check(e, state);
    if (model_ == nullptr || !is_valid(e)) {
        return alert;
    }
    if (alert) {
        alert->model_score = model_->score(compute_features(e, state)).score;
        ++scored_;
        return alert;
    }
    if (e.type == EventType::New) {
        return std::nullopt;  // cheapest part of the gate: no features needed
    }
    const auto row = compute_features(e, state);
    if (!passes_gate(e, row, model_config_)) {
        return std::nullopt;
    }
    ++scored_;
    const auto result = model_->score(row);
    if (!(result.score >= model_config_.threshold)) {
        return std::nullopt;
    }
    // Same cooldown as the rules: one model alert per pattern + trader + symbol per window.
    const auto key = model_cooldown_key(result.pattern, e.trader_id, e.symbol_id);
    const auto it = last_model_alert_ns_.find(key);
    if (it != last_model_alert_ns_.end() && e.timestamp_ns - it->second < cooldown_ns_) {
        return std::nullopt;
    }
    last_model_alert_ns_[key] = e.timestamp_ns;
    return Alert{.rule = result.pattern,
                 .window = Window::SixtySeconds,
                 .timestamp_ns = e.timestamp_ns,
                 .event_index = index,
                 .trader_id = e.trader_id,
                 .symbol_id = e.symbol_id,
                 .evidence = state.rolling.stats(e.trader_id, e.symbol_id, Window::SixtySeconds,
                                                 e.timestamp_ns),
                 .source = AlertSource::Model,
                 .model_score = result.score};
}

}  // namespace tsa
