#include "tsa/event_reader.hpp"

#include <ios>
#include <stdexcept>
#include <string>
#include <system_error>

namespace tsa {

EventReader::EventReader(const std::filesystem::path& path, std::size_t chunk_events)
    : path_(path) {
    if (chunk_events == 0) {
        throw std::invalid_argument("EventReader chunk size must be at least 1 event");
    }
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) {
        throw std::runtime_error("cannot read event file '" + path.string() + "': " + ec.message());
    }
    if (size % kEventSize != 0) {
        throw std::runtime_error("event file '" + path.string() + "' has " + std::to_string(size) +
                                 " bytes, which is not a multiple of " +
                                 std::to_string(kEventSize) + " (one event)");
    }
    in_.open(path, std::ios::binary);
    if (!in_) {
        throw std::runtime_error("cannot open event file '" + path.string() + "'");
    }
    total_events_ = size / kEventSize;
    buffer_.resize(chunk_events);
}

bool EventReader::refill() {
    in_.read(reinterpret_cast<char*>(buffer_.data()),
             static_cast<std::streamsize>(buffer_.size() * kEventSize));
    if (in_.bad()) {
        throw std::runtime_error("I/O error while reading event file '" + path_.string() + "'");
    }
    const auto bytes = static_cast<std::size_t>(in_.gcount());
    if (bytes % kEventSize != 0) {
        throw std::runtime_error("event file '" + path_.string() +
                                 "' ended in the middle of an event (was it changed while reading?)");
    }
    len_ = bytes / kEventSize;
    pos_ = 0;
    return len_ > 0;
}

bool EventReader::next(Event& out) {
    if (pos_ == len_ && !refill()) {
        return false;
    }
    out = buffer_[pos_++];
    if (!is_valid(out)) {
        throw std::runtime_error("event file '" + path_.string() + "': event #" +
                                 std::to_string(events_read_) +
                                 " has an invalid type, side or symbol_id (corrupt file?)");
    }
    if (out.timestamp_ns < last_timestamp_ns_) {
        throw std::runtime_error("event file '" + path_.string() + "': event #" +
                                 std::to_string(events_read_) +
                                 " has a timestamp earlier than the one before it");
    }
    last_timestamp_ns_ = out.timestamp_ns;
    ++events_read_;
    return true;
}

}  // namespace tsa
