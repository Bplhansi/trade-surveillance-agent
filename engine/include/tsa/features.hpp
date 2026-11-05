#pragma once

// Per-event feature rows for the anomaly model.
//
// One row describes one event plus the rolling 1 s and 60 s statistics of its trader + symbol,
// taken *after* the event has been applied (so only the past and present are used, never the
// future). IDs and timestamps are deliberately left out so the model can't memorise traders.
//
// The same function feeds both training (exported to a file, read by Python) and live scoring
//, so training and serving features can't drift apart.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string_view>

#include "tsa/engine_state.hpp"
#include "tsa/event.hpp"

namespace tsa {

inline constexpr std::size_t kNumFeatures = 33;
using FeatureRow = std::array<float, kNumFeatures>;

// Column names, in row order.
const std::array<std::string_view, kNumFeatures>& feature_names();

// Position of a named column. Throws std::out_of_range for an unknown name.
std::size_t feature_index(std::string_view name);

FeatureRow compute_features(const Event& e, const EngineState& state);

// Writes rows as raw little-endian float32 (<path>), plus <stem>.json describing the columns.
class FeatureWriter {
public:
    explicit FeatureWriter(const std::filesystem::path& path);
    ~FeatureWriter();  // finishes the .json description
    FeatureWriter(const FeatureWriter&) = delete;
    FeatureWriter& operator=(const FeatureWriter&) = delete;

    void write(const FeatureRow& row);
    std::uint64_t rows() const noexcept { return rows_; }

private:
    std::filesystem::path path_;
    std::ofstream out_;
    std::uint64_t rows_ = 0;
};

}  // namespace tsa
