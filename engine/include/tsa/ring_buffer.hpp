#pragma once

// A fixed-size ring of N slots, addressed by an ever-increasing index (for example
// "time bucket number"). Index i lives in slot i % N, so writing a new index overwrites
// the oldest one. Memory never grows.

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace tsa {

template <class T, std::size_t N>
class RingBuffer {
    static_assert(N > 0, "RingBuffer needs at least one slot");

public:
    static constexpr std::size_t capacity() noexcept { return N; }

    // The value for `index`, ready to update. If the slot still holds an older index,
    // it is reset to a fresh T{} first.
    T& slot(std::int64_t index) {
        Entry& e = entries_[position(index)];
        if (e.index != index) {
            e.index = index;
            e.value = T{};
        }
        return e.value;
    }

    // The value for `index`, or nullptr if that index was never written or was overwritten.
    const T* find(std::int64_t index) const noexcept {
        const Entry& e = entries_[position(index)];
        return e.index == index ? &e.value : nullptr;
    }

private:
    static constexpr std::int64_t kEmpty = std::numeric_limits<std::int64_t>::min();

    struct Entry {
        std::int64_t index = kEmpty;
        T value{};
    };

    static std::size_t position(std::int64_t index) noexcept {
        constexpr auto n = static_cast<std::int64_t>(N);
        return static_cast<std::size_t>(((index % n) + n) % n);
    }

    std::array<Entry, N> entries_{};
};

}  // namespace tsa
