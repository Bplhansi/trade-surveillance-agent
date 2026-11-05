#include "tsa/features.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace tsa {

namespace {

// Per window, in this order, with the suffix _1s or _60s.
constexpr std::size_t kPerWindow = 14;
constexpr std::size_t kEventFeatures = 5;
static_assert(kEventFeatures + 2 * kPerWindow == kNumFeatures);

void put_window(FeatureRow& row, std::size_t at, const WindowStats& s) {
    const double volume = static_cast<double>(s.buy_qty + s.sell_qty);
    const float values[kPerWindow] = {
        static_cast<float>(s.orders),
        static_cast<float>(s.cancels),
        static_cast<float>(s.cancel_ratio),
        static_cast<float>(s.avg_order_qty),
        static_cast<float>(s.max_order_qty),
        static_cast<float>(s.large_orders),
        static_cast<float>(s.large_cancels),
        static_cast<float>(s.large_cancel_ratio),
        static_cast<float>(s.buy_qty),
        static_cast<float>(s.sell_qty),
        static_cast<float>(volume > 0 ? static_cast<double>(s.net_qty) / volume : 0.0),
        static_cast<float>(s.trades),
        static_cast<float>(s.first_trade_price > 0 ? s.price_change / s.first_trade_price * 100.0
                                                   : 0.0),
        static_cast<float>(s.self_trades),
    };
    for (std::size_t i = 0; i < kPerWindow; ++i) {
        row[at + i] = values[i];
    }
}

}  // namespace

const std::array<std::string_view, kNumFeatures>& feature_names() {
    static const std::array<std::string_view, kNumFeatures> names = {
        "is_new", "is_cancel", "is_trade", "is_buy", "quantity",
        "orders_1s", "cancels_1s", "cancel_ratio_1s", "avg_order_qty_1s", "max_order_qty_1s",
        "large_orders_1s", "large_cancels_1s", "large_cancel_ratio_1s", "buy_qty_1s",
        "sell_qty_1s", "net_ratio_1s", "trades_1s", "price_change_pct_1s", "self_trades_1s",
        "orders_60s", "cancels_60s", "cancel_ratio_60s", "avg_order_qty_60s", "max_order_qty_60s",
        "large_orders_60s", "large_cancels_60s", "large_cancel_ratio_60s", "buy_qty_60s",
        "sell_qty_60s", "net_ratio_60s", "trades_60s", "price_change_pct_60s", "self_trades_60s",
    };
    return names;
}

std::size_t feature_index(std::string_view name) {
    const auto& names = feature_names();
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (names[i] == name) {
            return i;
        }
    }
    throw std::out_of_range("unknown feature '" + std::string(name) + "'");
}

FeatureRow compute_features(const Event& e, const EngineState& state) {
    FeatureRow row{};
    row[0] = e.type == EventType::New ? 1.0f : 0.0f;
    row[1] = e.type == EventType::Cancel ? 1.0f : 0.0f;
    row[2] = e.type == EventType::Trade ? 1.0f : 0.0f;
    row[3] = e.side == Side::Buy ? 1.0f : 0.0f;
    row[4] = static_cast<float>(e.quantity);
    put_window(row, kEventFeatures,
               state.rolling.stats(e.trader_id, e.symbol_id, Window::OneSecond, e.timestamp_ns));
    put_window(row, kEventFeatures + kPerWindow,
               state.rolling.stats(e.trader_id, e.symbol_id, Window::SixtySeconds, e.timestamp_ns));
    return row;
}

FeatureWriter::FeatureWriter(const std::filesystem::path& path) : path_(path) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    out_.open(path, std::ios::binary | std::ios::trunc);
    if (!out_) {
        throw std::runtime_error("cannot create features file '" + path.string() + "'");
    }
}

FeatureWriter::~FeatureWriter() {
    out_.close();
    auto meta_path = path_;
    meta_path.replace_extension(".json");
    std::ofstream meta(meta_path, std::ios::trunc);
    meta << "{\"rows\":" << rows_ << ",\"dtype\":\"float32\",\"columns\":[";
    const auto& names = feature_names();
    for (std::size_t i = 0; i < names.size(); ++i) {
        meta << (i ? "," : "") << '"' << names[i] << '"';
    }
    meta << "]}\n";
}

void FeatureWriter::write(const FeatureRow& row) {
    out_.write(reinterpret_cast<const char*>(row.data()),
               static_cast<std::streamsize>(row.size() * sizeof(float)));
    if (!out_) {
        throw std::runtime_error("failed writing features file '" + path_.string() + "'");
    }
    ++rows_;
}

}  // namespace tsa
