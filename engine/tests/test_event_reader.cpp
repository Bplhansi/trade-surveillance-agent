// Tests for the streaming event file reader.
#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "test_helpers.hpp"
#include "tsa/event.hpp"
#include "tsa/event_reader.hpp"

using tsa_test::make_sequence;
using tsa_test::write_temp;

namespace {

std::vector<tsa::Event> read_all(tsa::EventReader& reader) {
    std::vector<tsa::Event> out;
    for (const tsa::Event& e : reader) {
        out.push_back(e);
    }
    return out;
}

}  // namespace

TEST(EventReader, ReadsEveryEventInFileOrder) {
    const auto events = make_sequence(10);
    tsa::EventReader reader(write_temp("reader_all.bin", events));

    EXPECT_EQ(reader.total_events(), 10u);
    EXPECT_EQ(read_all(reader), events);
    EXPECT_EQ(reader.events_read(), 10u);
}

TEST(EventReader, HandlesEventsAcrossChunkBoundaries) {
    const auto events = make_sequence(10);
    tsa::EventReader reader(write_temp("reader_chunks.bin", events), /*chunk_events=*/3);

    EXPECT_EQ(read_all(reader), events);
}

TEST(EventReader, MatchesReadEventsOnThePythonFile) {
    tsa::EventReader reader(TSA_PY_EVENTS_FILE);

    EXPECT_EQ(read_all(reader), tsa::read_events(TSA_PY_EVENTS_FILE));
}

TEST(EventReader, EmptyFileGivesNoEvents) {
    tsa::EventReader reader(write_temp("reader_empty.bin", {}));

    EXPECT_TRUE(read_all(reader).empty());
}

TEST(EventReader, ThrowsWhenFileIsMissing) {
    EXPECT_THROW(tsa::EventReader(tsa_test::temp_path("reader_missing.bin")), std::runtime_error);
}

TEST(EventReader, ThrowsWhenFileHasAPartialEvent) {
    const auto path = write_temp("reader_partial.bin", make_sequence(2));
    std::filesystem::resize_file(path, 79);  // 1 full event + 39 stray bytes

    EXPECT_THROW(tsa::EventReader{path}, std::runtime_error);
}

TEST(EventReader, ThrowsWhenTimestampsGoBackwards) {
    auto events = make_sequence(3);
    std::swap(events[1].timestamp_ns, events[2].timestamp_ns);
    tsa::EventReader reader(write_temp("reader_backwards.bin", events));

    EXPECT_THROW(read_all(reader), std::runtime_error);
}

TEST(EventReader, ThrowsOnAnEventWithAnInvalidType) {
    auto events = make_sequence(3);
    events[1].type = static_cast<tsa::EventType>(7);
    tsa::EventReader reader(write_temp("reader_bad_type.bin", events));

    EXPECT_THROW(read_all(reader), std::runtime_error);
}

TEST(EventReader, ThrowsOnAnEventWithAHugeSymbolId) {
    auto events = make_sequence(3);
    events[2].symbol_id = 0xFFFFFFFF;
    tsa::EventReader reader(write_temp("reader_bad_symbol.bin", events));

    EXPECT_THROW(read_all(reader), std::runtime_error);
}

TEST(EventReader, ChunkSizeThatDividesTheFileExactly) {
    const auto events = make_sequence(10);
    tsa::EventReader reader(write_temp("reader_exact_chunks.bin", events), 5);

    EXPECT_EQ(read_all(reader), events);
}

TEST(EventReader, RejectsAZeroChunkSize) {
    const auto path = write_temp("reader_zero_chunk.bin", make_sequence(1));

    EXPECT_THROW(tsa::EventReader(path, 0), std::invalid_argument);
}
