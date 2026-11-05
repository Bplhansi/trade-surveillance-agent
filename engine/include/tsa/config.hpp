#pragma once

// A tiny reader for .ini-style config files:
//
//   ; comment            # also a comment
//   [section]
//   key = value          ; trailing comments are allowed
//
// Values are read as numbers on request. Errors name the file and line.

#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace tsa {

class Config {
public:
    // Throw std::runtime_error on unreadable files or malformed lines.
    static Config from_file(const std::filesystem::path& path);
    static Config from_string(std::string_view text, std::string source = "<config>");

    bool has(std::string_view section, std::string_view key) const;

    // The value as a number, or `fallback` if the key is missing.
    // Throws std::runtime_error if the value is present but not a number.
    double get_double(std::string_view section, std::string_view key, double fallback) const;

    // Every "section.key" in the file, sorted.
    std::vector<std::string> keys() const;

    const std::string& source() const noexcept { return source_; }

private:
    std::string source_;
    std::map<std::string, std::string, std::less<>> values_;  // "section.key" -> raw value
};

}  // namespace tsa
