#include "tsa/alert_writer.hpp"

#include <array>
#include <charconv>
#include <stdexcept>
#include <string_view>

namespace tsa {

namespace {

// Shortest text that reads back as exactly the same double (e.g. 1.0 -> "1", 0.8 -> "0.8").
std::string number(double value) {
    std::array<char, 32> buf{};
    const auto [end, ec] = std::to_chars(buf.data(), buf.data() + buf.size(), value);
    return ec == std::errc{} ? std::string(buf.data(), end) : std::string("null");
}

template <class Int>
std::string number(Int value) {
    return std::to_string(value);
}

class JsonObject {
public:
    template <class T>
    JsonObject& field(std::string_view name, const T& value) {
        key(name);
        text_ += number(value);
        return *this;
    }
    JsonObject& text(std::string_view name, std::string_view value) {  // value: plain ASCII only
        key(name);
        text_ += '"';
        text_ += value;
        text_ += '"';
        return *this;
    }
    JsonObject& raw(std::string_view name, const std::string& json) {
        key(name);
        text_ += json;
        return *this;
    }
    std::string done() { return text_ + "}"; }

private:
    void key(std::string_view name) {
        text_ += first_ ? "" : ",";
        first_ = false;
        text_ += '"';
        text_ += name;
        text_ += "\":";
    }
    std::string text_ = "{";
    bool first_ = true;
};

}  // namespace

std::string to_json(const Alert& a, std::uint64_t alert_id) {
    const auto& s = a.evidence;
    const std::string evidence = JsonObject{}
                                     .field("orders", s.orders)
                                     .field("cancels", s.cancels)
                                     .field("cancel_ratio", s.cancel_ratio)
                                     .field("avg_order_qty", s.avg_order_qty)
                                     .field("max_order_qty", s.max_order_qty)
                                     .field("large_orders", s.large_orders)
                                     .field("large_cancels", s.large_cancels)
                                     .field("large_cancel_ratio", s.large_cancel_ratio)
                                     .field("buy_qty", s.buy_qty)
                                     .field("sell_qty", s.sell_qty)
                                     .field("net_qty", s.net_qty)
                                     .field("trades", s.trades)
                                     .field("first_trade_price", s.first_trade_price)
                                     .field("price_change", s.price_change)
                                     .field("self_trades", s.self_trades)
                                     .done();
    return JsonObject{}
        .field("alert_id", alert_id)
        .text("rule", rule_name(a.rule))
        .field("timestamp_ns", a.timestamp_ns)
        .field("event_index", a.event_index)
        .field("trader_id", a.trader_id)
        .field("symbol_id", a.symbol_id)
        .text("window", a.window == Window::OneSecond ? "1s" : "60s")
        .text("source", a.source == AlertSource::Model ? "model" : "rules")
        .raw("model_score", a.model_score ? number(*a.model_score) : std::string("null"))
        .raw("evidence", evidence)
        .done();
}

AlertWriter::AlertWriter(const std::filesystem::path& path) : path_(path) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    out_.open(path, std::ios::trunc);
    if (!out_) {
        throw std::runtime_error("cannot create alerts file '" + path.string() + "'");
    }
}

void AlertWriter::write(const Alert& alert) {
    out_ << to_json(alert, ++written_) << '\n';
    if (!out_) {
        throw std::runtime_error("failed writing alerts file '" + path_.string() + "'");
    }
}

}  // namespace tsa
