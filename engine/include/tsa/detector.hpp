#pragma once

// Runs the rules and (optionally) the ONNX model on each event.
//
// 1. The rules run first. If one fires, its alert also gets the model's score.
// 2. Otherwise, events that pass a cheap gate (see passes_gate) are scored by the model;
//    a score at or above [model] threshold raises a "model" alert. New-order events and
//    quiet traders are never scored, which keeps the average cost per event low.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <unordered_map>

#include "tsa/engine_state.hpp"
#include "tsa/features.hpp"
#include "tsa/model.hpp"
#include "tsa/rules.hpp"

namespace tsa {

// Cheap first check: a Cancel after a large cancel in the last second, or a Trade by a trader
// with a few trades in the last minute.
bool passes_gate(const Event& e, const FeatureRow& row, const ModelConfig& config) noexcept;

// Key for the model's cooldown: one alert per pattern + trader + symbol per window.
// Kept as separate fields: packing all three into 64 bits made large trader IDs collide.
struct ModelCooldownKey {
    Rule pattern;
    std::uint32_t trader_id;
    std::uint32_t symbol_id;
    bool operator==(const ModelCooldownKey&) const = default;
};

struct ModelCooldownKeyHash {
    std::size_t operator()(const ModelCooldownKey& k) const noexcept {
        const auto pair = (static_cast<std::uint64_t>(k.trader_id) << 32) | k.symbol_id;
        return std::hash<std::uint64_t>{}(pair) ^ (static_cast<std::size_t>(k.pattern) << 1);
    }
};

ModelCooldownKey model_cooldown_key(Rule pattern, std::uint32_t trader_id,
                                    std::uint32_t symbol_id) noexcept;

class Detector {
public:
    // `model` may be null (rules only). It is not owned and must outlive the detector.
    Detector(RulesConfig config, ModelScorer* model);

    // Call once per event, in file order, after the state has been updated with it.
    std::optional<Alert> check(const Event& e, const EngineState& state);

    // How many events were sent to the model.
    std::uint64_t scored() const noexcept { return scored_; }

private:
    Rules rules_;
    ModelScorer* model_;
    ModelConfig model_config_;
    std::uint64_t cooldown_ns_;
    std::uint64_t event_index_ = 0;
    std::uint64_t scored_ = 0;
    std::unordered_map<ModelCooldownKey, std::uint64_t, ModelCooldownKeyHash>
        last_model_alert_ns_;  // last model alert time per pattern + trader + symbol
};

}  // namespace tsa
