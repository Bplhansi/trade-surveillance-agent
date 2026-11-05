#include "tsa/event.hpp"

#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <system_error>

namespace tsa {

std::vector<Event> read_events(const std::filesystem::path& path) {
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

    std::vector<Event> events(size / kEventSize);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open event file '" + path.string() + "'");
    }
    // Event is trivially copyable with a fixed layout, so the file bytes map straight onto it.
    in.read(reinterpret_cast<char*>(events.data()), static_cast<std::streamsize>(size));
    if (!in) {
        throw std::runtime_error("failed to read all of event file '" + path.string() + "'");
    }
    return events;
}

}  // namespace tsa
