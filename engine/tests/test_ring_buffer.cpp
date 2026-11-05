// Tests for the fixed-size ring buffer of time slots.
#include <gtest/gtest.h>

#include "tsa/ring_buffer.hpp"

TEST(RingBuffer, HasAFixedCapacity) {
    EXPECT_EQ((tsa::RingBuffer<int, 4>::capacity()), 4u);
}

TEST(RingBuffer, StartsWithEverySlotEmpty) {
    const tsa::RingBuffer<int, 3> ring;

    EXPECT_EQ(ring.find(0), nullptr);
    EXPECT_EQ(ring.find(2), nullptr);
}

TEST(RingBuffer, StoresAValueForASlotIndex) {
    tsa::RingBuffer<int, 3> ring;

    ring.slot(5) = 42;

    ASSERT_NE(ring.find(5), nullptr);
    EXPECT_EQ(*ring.find(5), 42);
}

TEST(RingBuffer, NewerIndexOverwritesTheOldestEntryInTheSameSlot) {
    tsa::RingBuffer<int, 3> ring;
    ring.slot(0) = 10;
    ring.slot(1) = 11;
    ring.slot(2) = 12;

    ring.slot(3) = 13;  // same physical slot as index 0

    EXPECT_EQ(ring.find(0), nullptr);  // the old value is gone
    EXPECT_EQ(*ring.find(1), 11);
    EXPECT_EQ(*ring.find(3), 13);
}

TEST(RingBuffer, ReusedSlotStartsFromAFreshValue) {
    tsa::RingBuffer<int, 2> ring;
    ring.slot(0) = 99;

    int& reused = ring.slot(2);  // overwrites index 0

    EXPECT_EQ(reused, 0);
}

TEST(RingBuffer, SameIndexReturnsTheSameValueToUpdate) {
    tsa::RingBuffer<int, 2> ring;

    ring.slot(7) += 1;
    ring.slot(7) += 1;

    EXPECT_EQ(*ring.find(7), 2);
}
