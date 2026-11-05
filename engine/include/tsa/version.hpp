#pragma once

#include <string_view>

namespace tsa {

// Returns the version of the detection engine, e.g. "0.1.0".
std::string_view version() noexcept;

}  // namespace tsa
