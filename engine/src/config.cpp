#include "tsa/config.hpp"

#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace tsa {

namespace {

std::string_view trim(std::string_view s) {
    const auto first = s.find_first_not_of(" \t\r");
    if (first == std::string_view::npos) {
        return {};
    }
    const auto last = s.find_last_not_of(" \t\r");
    return s.substr(first, last - first + 1);
}

std::string_view strip_comment(std::string_view line) {
    const auto pos = line.find_first_of(";#");
    return pos == std::string_view::npos ? line : line.substr(0, pos);
}

std::string full_key(std::string_view section, std::string_view key) {
    std::string out(section);
    out += '.';
    out += key;
    return out;
}

}  // namespace

Config Config::from_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open config file '" + path.string() + "'");
    }
    std::stringstream text;
    text << in.rdbuf();
    return from_string(text.str(), path.string());
}

Config Config::from_string(std::string_view text, std::string source) {
    Config cfg;
    cfg.source_ = std::move(source);
    std::string section;
    std::size_t line_no = 0;

    while (!text.empty()) {
        const auto end = text.find('\n');
        const auto raw = text.substr(0, end);
        text = end == std::string_view::npos ? std::string_view{} : text.substr(end + 1);
        ++line_no;

        const auto line = trim(strip_comment(raw));
        if (line.empty()) {
            continue;
        }
        const auto where = cfg.source_ + ":" + std::to_string(line_no);
        if (line.front() == '[') {
            if (line.back() != ']' || line.size() < 3) {
                throw std::runtime_error(where + ": malformed section header '" + std::string(line) + "'");
            }
            section = std::string(trim(line.substr(1, line.size() - 2)));
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string_view::npos) {
            throw std::runtime_error(where + ": expected 'key = value', got '" + std::string(line) + "'");
        }
        if (section.empty()) {
            throw std::runtime_error(where + ": setting must be inside a [section]");
        }
        const auto key = trim(line.substr(0, eq));
        if (key.empty()) {
            throw std::runtime_error(where + ": missing key before '='");
        }
        cfg.values_[full_key(section, key)] = std::string(trim(line.substr(eq + 1)));
    }
    return cfg;
}

bool Config::has(std::string_view section, std::string_view key) const {
    return values_.contains(full_key(section, key));
}

double Config::get_double(std::string_view section, std::string_view key, double fallback) const {
    const auto name = full_key(section, key);
    const auto it = values_.find(name);
    if (it == values_.end()) {
        return fallback;
    }
    const std::string& text = it->second;
    double value = 0.0;
    const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || ptr != text.data() + text.size()) {
        throw std::runtime_error(source_ + ": " + name + " = '" + text + "' is not a number");
    }
    return value;
}

std::vector<std::string> Config::keys() const {
    std::vector<std::string> out;
    out.reserve(values_.size());
    for (const auto& [name, value] : values_) {
        out.push_back(name);
    }
    return out;
}

}  // namespace tsa
