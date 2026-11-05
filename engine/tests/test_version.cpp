// First test for the engine. It proves the build, the test library (GoogleTest)
// and CTest are all wired together correctly.
#include <gtest/gtest.h>

#include "tsa/version.hpp"

TEST(Version, ReturnsCurrentEngineVersion) {
    EXPECT_EQ(tsa::version(), "0.1.0");
}
