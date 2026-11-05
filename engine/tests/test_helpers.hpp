#pragma once

// Shared helpers for the engine tests: build events and write small event files.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tsa/event.hpp"

namespace tsa {
// Lets GoogleTest print an Event readably when a test fails.
inline void PrintTo(const Event& e, std::ostream* os) {
    *os << "{ts=" << e.timestamp_ns << " order=" << e.order_id << " trader=" << e.trader_id
        << " symbol=" << e.symbol_id << " price=" << e.price << " qty=" << e.quantity
        << " type=" << static_cast<int>(e.type) << " side=" << static_cast<int>(e.side) << "}";
}
}  // namespace tsa

namespace tsa_test {

inline tsa::Event make_event(std::uint64_t ts, std::uint32_t order_id, std::uint32_t trader_id,
                             std::uint32_t symbol_id, double price, std::uint32_t quantity,
                             tsa::EventType type, tsa::Side side) {
    return tsa::Event{.timestamp_ns = ts, .order_id = order_id, .trader_id = trader_id,
                      .symbol_id = symbol_id, .price = price, .quantity = quantity,
                      .type = type, .side = side};
}

// n simple events, one per microsecond, cycling through New / Cancel / Trade.
inline std::vector<tsa::Event> make_sequence(std::size_t n) {
    std::vector<tsa::Event> events;
    for (std::uint32_t i = 0; i < n; ++i) {
        events.push_back(make_event(1'000'000 + i * 1'000ULL, i + 1, 1 + i % 4, 1 + i % 2,
                                    50.0 + 0.01 * i, 10 * (i + 1),
                                    static_cast<tsa::EventType>(i % 3),
                                    static_cast<tsa::Side>(i % 2)));
    }
    return events;
}

inline std::filesystem::path temp_path(const std::string& name) {
    return std::filesystem::temp_directory_path() / ("tsa_test_" + name);
}

// Writes events as raw 40-byte records (the shared format) and returns the path.
inline std::filesystem::path write_temp(const std::string& name,
                                        const std::vector<tsa::Event>& events) {
    const auto path = temp_path(name);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot write " + path.string());
    }
    out.write(reinterpret_cast<const char*>(events.data()),
              static_cast<std::streamsize>(events.size() * sizeof(tsa::Event)));
    return path;
}

}  // namespace tsa_test
