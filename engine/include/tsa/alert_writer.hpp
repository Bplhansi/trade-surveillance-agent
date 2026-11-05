#pragma once

// Writes alerts as JSON Lines: one JSON object per line, numbered from 1.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include "tsa/rules.hpp"

namespace tsa {

// One alert as a single-line JSON object.
std::string to_json(const Alert& alert, std::uint64_t alert_id);

class AlertWriter {
public:
    // Creates missing parent folders and replaces any existing file.
    // Throws std::runtime_error if the file can't be created.
    explicit AlertWriter(const std::filesystem::path& path);

    void write(const Alert& alert);
    std::uint64_t written() const noexcept { return written_; }

private:
    std::filesystem::path path_;
    std::ofstream out_;
    std::uint64_t written_ = 0;
};

}  // namespace tsa
