#pragma once

// Streams events from a binary event file in order, a chunk at a time,
// so large files don't have to fit in memory at once.
//
//   tsa::EventReader reader("data/train.events.bin");
//   for (const tsa::Event& e : reader) { ... }

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include "tsa/event.hpp"

namespace tsa {

class EventReader {
public:
    static constexpr std::size_t kDefaultChunkEvents = 64 * 1024 / sizeof(Event);  // 64 KB

    // Throws std::runtime_error if the file can't be opened or holds a partial event,
    // and std::invalid_argument if chunk_events is 0.
    explicit EventReader(const std::filesystem::path& path,
                         std::size_t chunk_events = kDefaultChunkEvents);

    // Copies the next event into `out`. Returns false at the end of the file.
    // Throws std::runtime_error if timestamps go backwards.
    bool next(Event& out);

    std::uint64_t total_events() const noexcept { return total_events_; }
    std::uint64_t events_read() const noexcept { return events_read_; }

    // Lets the reader be used in a range-based for loop (single pass).
    struct Sentinel {};
    class Iterator {
    public:
        explicit Iterator(EventReader& reader) : reader_(&reader) { ++*this; }
        const Event& operator*() const noexcept { return current_; }
        Iterator& operator++() {
            done_ = !reader_->next(current_);
            return *this;
        }
        void operator++(int) { ++*this; }
        bool operator==(Sentinel) const noexcept { return done_; }

    private:
        EventReader* reader_;
        Event current_{};
        bool done_ = false;
    };
    Iterator begin() { return Iterator(*this); }
    Sentinel end() const noexcept { return {}; }

private:
    bool refill();

    std::filesystem::path path_;
    std::ifstream in_;
    std::vector<Event> buffer_;
    std::size_t pos_ = 0;
    std::size_t len_ = 0;
    std::uint64_t total_events_ = 0;
    std::uint64_t events_read_ = 0;
    std::uint64_t last_timestamp_ns_ = 0;
};

}  // namespace tsa
