// Tests for writing alerts as JSON Lines.
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "test_helpers.hpp"
#include "tsa/alert_writer.hpp"

namespace {

tsa::Alert sample_alert() {
    tsa::Alert a;
    a.rule = tsa::Rule::Spoofing;
    a.window = tsa::Window::OneSecond;
    a.timestamp_ns = 1767605400123456789ULL;
    a.event_index = 5123;
    a.trader_id = 42;
    a.symbol_id = 3;
    a.evidence.orders = 4;
    a.evidence.cancels = 4;
    a.evidence.cancel_ratio = 1.0;
    a.evidence.avg_order_qty = 2512.5;
    a.evidence.max_order_qty = 3000;
    a.evidence.large_orders = 4;
    a.evidence.large_cancels = 4;
    a.evidence.large_cancel_ratio = 1.0;
    a.evidence.buy_qty = 0;
    a.evidence.sell_qty = 120;
    a.evidence.net_qty = -120;
    a.evidence.trades = 1;
    a.evidence.first_trade_price = 101.25;
    a.evidence.price_change = 0.0;
    a.evidence.self_trades = 0;
    return a;
}

std::string read_file(const std::filesystem::path& p) {
    std::ifstream in(p);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

}  // namespace

TEST(AlertJson, IsOneLineWithEveryField) {
    EXPECT_EQ(tsa::to_json(sample_alert(), 1),
              R"({"alert_id":1,"rule":"spoofing","timestamp_ns":1767605400123456789,)"
              R"("event_index":5123,"trader_id":42,"symbol_id":3,"window":"1s",)"
              R"("source":"rules","model_score":null,)"
              R"("evidence":{"orders":4,"cancels":4,"cancel_ratio":1,"avg_order_qty":2512.5,)"
              R"("max_order_qty":3000,"large_orders":4,"large_cancels":4,"large_cancel_ratio":1,)"
              R"("buy_qty":0,"sell_qty":120,"net_qty":-120,"trades":1,"first_trade_price":101.25,)"
              R"("price_change":0,"self_trades":0}})");
}

TEST(AlertJson, ModelAlertsNameTheirSourceAndScore) {
    auto a = sample_alert();
    a.source = tsa::AlertSource::Model;
    a.model_score = 7.25;

    const auto json = tsa::to_json(a, 3);

    EXPECT_NE(json.find(R"("source":"model","model_score":7.25,)"), std::string::npos) << json;
}

TEST(AlertWriter, WritesOneNumberedLinePerAlert) {
    const auto path = tsa_test::temp_path("alerts_dir") / "nested" / "alerts.jsonl";
    std::filesystem::remove_all(tsa_test::temp_path("alerts_dir"));
    {
        tsa::AlertWriter writer(path);  // creates missing folders
        writer.write(sample_alert());
        writer.write(sample_alert());
        EXPECT_EQ(writer.written(), 2u);
    }

    const auto text = read_file(path);
    EXPECT_EQ(text, tsa::to_json(sample_alert(), 1) + "\n" + tsa::to_json(sample_alert(), 2) + "\n");
}
