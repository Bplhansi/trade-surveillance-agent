# Build instructions for the C++ detection engine.
cmake_minimum_required(VERSION 3.20)
project(trade_surveillance_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Single-config generators (Makefiles/Ninja on Linux) build unoptimised if no type is
# given; default to Release so benchmark numbers are never accidentally from a slow build.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Optimisation -----------------------------------------------------------
# Release builds use the compiler's top optimisation level, set explicitly:
# -O3 for GCC/Clang (Linux), /O2 for MSVC (Windows has no -O3).
if(MSVC)
    set(TSA_OPT_FLAG "/O2")
    set(TSA_WARNINGS /W4 /permissive-)
else()
    set(TSA_OPT_FLAG "-O3")
    set(TSA_WARNINGS -Wall -Wextra -Wpedantic)
endif()
add_compile_options($<$<CONFIG:Release>:${TSA_OPT_FLAG}>)

# --- Engine library: the real code lives here -------------------------------
add_library(tsa_core
    src/alert_writer.cpp
    src/features.cpp
    src/detector.cpp
    src/model.cpp
    src/config.cpp
    src/rules.cpp
    src/event.cpp
    src/event_reader.cpp
    src/latency.cpp
    src/market_state.cpp
    src/rolling_stats.cpp
    src/version.cpp
)
target_include_directories(tsa_core PUBLIC include)
target_compile_options(tsa_core PRIVATE ${TSA_WARNINGS})

# --- ONNX Runtime: runs the trained model inside the engine ------------
# Downloads Microsoft's prebuilt release, pinned by version and SHA-256 so a changed file
# fails the build. Configure with -DTSA_WITH_ONNX=OFF to build without model scoring.
include(FetchContent)
option(TSA_WITH_ONNX "Score events with an ONNX model via ONNX Runtime" ON)
set(TSA_ORT_DLLS "")
if(TSA_WITH_ONNX)
    set(TSA_ORT_VERSION 1.31.1)
    set(_ort_base "https://github.com/microsoft/onnxruntime/releases/download/v${TSA_ORT_VERSION}")
    if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(_ort_url "${_ort_base}/onnxruntime-win-x64-${TSA_ORT_VERSION}.zip")
        set(_ort_sha 5869e26bb25314375f2d5ef1877e3ea06b1543d6c710e6e4a1ec2f7177ab4b2c)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(_ort_url "${_ort_base}/onnxruntime-linux-x64-${TSA_ORT_VERSION}.tgz")
        set(_ort_sha c849e36a1ef3e22d4ff217b9d1ce4a3cf5b9467b975610913ae959ad5b16bdc9)
    else()
        message(FATAL_ERROR "No prebuilt ONNX Runtime for this platform; use -DTSA_WITH_ONNX=OFF")
    endif()
    FetchContent_Declare(onnxruntime URL "${_ort_url}" URL_HASH SHA256=${_ort_sha}
                         DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(onnxruntime)  # just unpacks: the release has no CMakeLists
    add_library(onnxruntime_prebuilt SHARED IMPORTED)
    set_target_properties(onnxruntime_prebuilt PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include")
    if(WIN32)
        set_target_properties(onnxruntime_prebuilt PROPERTIES
            IMPORTED_IMPLIB "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib"
            IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll")
        # Windows has an older onnxruntime.dll in System32; ours must sit next to each program.
        set(TSA_ORT_DLLS "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll"
                         "${onnxruntime_SOURCE_DIR}/lib/onnxruntime_providers_shared.dll")
    else()
        set_target_properties(onnxruntime_prebuilt PROPERTIES
            IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so")
    endif()
    target_link_libraries(tsa_core PUBLIC onnxruntime_prebuilt)
    target_compile_definitions(tsa_core PUBLIC TSA_WITH_ONNX=1)
endif()

# Copies the ONNX Runtime DLLs next to a program on Windows (does nothing elsewhere).
function(tsa_copy_runtime_dlls target)
    if(TSA_ORT_DLLS)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TSA_ORT_DLLS} $<TARGET_FILE_DIR:${target}>)
    endif()
endfunction()

# --- Programs ---------------------------------------------------------------
add_executable(tsa_replay src/replay_main.cpp)
target_link_libraries(tsa_replay PRIVATE tsa_core)
target_compile_options(tsa_replay PRIVATE ${TSA_WARNINGS})
target_compile_definitions(tsa_replay PRIVATE
    TSA_BUILD_CONFIG="$<CONFIG>"
    TSA_BUILD_OPT="$<$<CONFIG:Release>:${TSA_OPT_FLAG}>")
tsa_copy_runtime_dlls(tsa_replay)

# --- Tests ------------------------------------------------------------------
enable_testing()

# Download GoogleTest at configure time (pinned to a fixed release).
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# On Windows, make GoogleTest use the same runtime library as our code.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(tsa_tests
    tests/test_alert_writer.cpp
    tests/test_features.cpp
    tests/test_model.cpp
    tests/test_config.cpp
    tests/test_rules.cpp
    tests/test_event.cpp
    tests/test_event_reader.cpp
    tests/test_latency.cpp
    tests/test_market_state.cpp
    tests/test_replay.cpp
    tests/test_ring_buffer.cpp
    tests/test_rolling_stats.cpp
    tests/test_version.cpp
)
target_link_libraries(tsa_tests PRIVATE tsa_core GTest::gtest_main)
target_compile_options(tsa_tests PRIVATE ${TSA_WARNINGS})
tsa_copy_runtime_dlls(tsa_tests)

# Round-trip check: before the C++ tests run, Python writes 10 sample events
# to this file, and C++ tests read them back.
find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
set(TSA_PY_EVENTS_FILE "${CMAKE_BINARY_DIR}/roundtrip/python_events.bin")
target_compile_definitions(tsa_tests PRIVATE
    TSA_PY_EVENTS_FILE="${TSA_PY_EVENTS_FILE}"
    TSA_RULES_FILE="${PROJECT_SOURCE_DIR}/../config/rules.ini")

add_test(
    NAME python_writes_sample_events
    COMMAND Python3::Interpreter -m sim.write_sample_events "${TSA_PY_EVENTS_FILE}"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/.."
)
set_tests_properties(python_writes_sample_events PROPERTIES FIXTURES_SETUP python_events)

# Model parity check: Python writes a tiny ONNX model, feature rows and the scores
# scikit-learn gives them; C++ tests must reproduce the scores. Needs requirements.txt
# installed for this Python (locally: configure with -DPython3_EXECUTABLE=.venv/...).
set(TSA_PARITY_DIR "${CMAKE_BINARY_DIR}/parity")
target_compile_definitions(tsa_tests PRIVATE TSA_PARITY_DIR="${TSA_PARITY_DIR}")
set(TSA_TEST_FIXTURES python_events)
if(TSA_WITH_ONNX)
    add_test(
        NAME python_exports_parity_model
        COMMAND Python3::Interpreter -m ml.export_onnx --parity-dir "${TSA_PARITY_DIR}"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/.."
    )
    set_tests_properties(python_exports_parity_model PROPERTIES FIXTURES_SETUP parity_model)
    list(APPEND TSA_TEST_FIXTURES parity_model)
endif()

include(GoogleTest)
# PRE_TEST: list the tests when CTest runs, after the ONNX Runtime DLLs have been copied.
gtest_discover_tests(tsa_tests DISCOVERY_MODE PRE_TEST
                     PROPERTIES FIXTURES_REQUIRED "${TSA_TEST_FIXTURES}")

# Smoke tests for the replay program itself.
add_test(NAME tsa_replay_prints_latency
         COMMAND tsa_replay "${TSA_PY_EVENTS_FILE}" --repeat 2)
set_tests_properties(tsa_replay_prints_latency PROPERTIES
    FIXTURES_REQUIRED python_events
    PASS_REGULAR_EXPRESSION "throughput.*events/s.*latency p50.*ns.*latency p99.*ns")
add_test(NAME tsa_replay_rejects_missing_file COMMAND tsa_replay "does-not-exist.bin")
set_tests_properties(tsa_replay_rejects_missing_file PROPERTIES
    PASS_REGULAR_EXPRESSION "error: cannot read event file")
add_test(NAME tsa_replay_rejects_bad_speed COMMAND tsa_replay "x.bin" --speed abc)
set_tests_properties(tsa_replay_rejects_bad_speed PROPERTIES
    PASS_REGULAR_EXPRESSION "error: --speed.*usage: tsa_replay")
add_test(NAME tsa_replay_rejects_bad_repeat COMMAND tsa_replay "x.bin" --repeat 2.5)
set_tests_properties(tsa_replay_rejects_bad_repeat PROPERTIES
    PASS_REGULAR_EXPRESSION "error: --repeat.*usage: tsa_replay")
add_test(NAME tsa_replay_writes_alerts
         COMMAND tsa_replay "${TSA_PY_EVENTS_FILE}"
                 --rules "${PROJECT_SOURCE_DIR}/../config/rules.ini"
                 --alerts "${CMAKE_BINARY_DIR}/smoke/alerts.jsonl")
set_tests_properties(tsa_replay_writes_alerts PROPERTIES
    FIXTURES_REQUIRED python_events
    PASS_REGULAR_EXPRESSION "rules: .*rules.ini.*alerts written to: .*alerts.jsonl")
add_test(NAME tsa_replay_rejects_bad_rules_file
         COMMAND tsa_replay "${TSA_PY_EVENTS_FILE}" --rules "no-such-rules.ini")
set_tests_properties(tsa_replay_rejects_bad_rules_file PROPERTIES
    FIXTURES_REQUIRED python_events
    PASS_REGULAR_EXPRESSION "error: cannot open config file")
add_test(NAME tsa_replay_writes_features
         COMMAND tsa_replay "${TSA_PY_EVENTS_FILE}" --features "${CMAKE_BINARY_DIR}/smoke/sample.features.bin")
set_tests_properties(tsa_replay_writes_features PROPERTIES
    FIXTURES_REQUIRED python_events
    PASS_REGULAR_EXPRESSION "features written to: .*sample.features.bin")
add_test(NAME tsa_replay_reports_build_type
         COMMAND tsa_replay "${TSA_PY_EVENTS_FILE}")
set_tests_properties(tsa_replay_reports_build_type PROPERTIES
    FIXTURES_REQUIRED python_events
    PASS_REGULAR_EXPRESSION "build: (Release|Debug|RelWithDebInfo|MinSizeRel)")
