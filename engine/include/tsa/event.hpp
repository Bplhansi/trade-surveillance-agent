#pragma once

// The trading event: the one data format every part of the system shares.
// It must match sim/event_format.py byte for byte (40-byte little-endian record).

#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <vector>

namespace tsa {

enum class EventType : std::uint8_t { New = 0, Cancel = 1, Trade = 2 };
enum class Side : std::uint8_t { Buy = 0, Sell = 1 };

struct Event {
    std::uint64_t timestamp_ns;
    std::uint32_t order_id;
    std::uint32_t trader_id;
    std::uint32_t symbol_id;
    double price;
    std::uint32_t quantity;
    EventType type;
    Side side;

    bool operator==(const Event&) const = default;
};

// Compile-time guards: the build fails if the layout ever drifts from the Python format.
static_assert(std::endian::native == std::endian::little, "event files are little-endian");
static_assert(std::is_trivially_copyable_v<Event>, "Event must be copyable as raw bytes");
static_assert(sizeof(Event) == 40, "Event must be 40 bytes to match sim/event_format.py");
static_assert(offsetof(Event, price) == 24 && offsetof(Event, quantity) == 32 &&
                  offsetof(Event, type) == 36 && offsetof(Event, side) == 37,
              "Event field offsets must match sim/event_format.py");

inline constexpr std::size_t kEventSize = sizeof(Event);

// Largest symbol_id the engine accepts. Keeps per-symbol tables small even if a file is corrupt.
inline constexpr std::uint32_t kMaxSymbolId = 65'535;

// True if every field holds a value the engine can safely use. Events come straight from
// file bytes, so anything (e.g. type = 7) is possible in a corrupt or foreign file.
constexpr bool is_valid(const Event& e) noexcept {
    return static_cast<std::uint8_t>(e.type) <= static_cast<std::uint8_t>(EventType::Trade) &&
           static_cast<std::uint8_t>(e.side) <= static_cast<std::uint8_t>(Side::Sell) &&
           e.symbol_id >= 1 && e.symbol_id <= kMaxSymbolId;
}

// Reads every event from a binary event file.
// Throws std::runtime_error if the file can't be opened or holds a partial event.
std::vector<Event> read_events(const std::filesystem::path& path);

}  // namespace tsa
