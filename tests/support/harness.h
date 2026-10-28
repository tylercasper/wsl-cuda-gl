// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

#pragma once
// A deliberately tiny test harness, so the tests have no dependencies.
//
// A test binary exits 0 (pass), 1 (fail) or 77 (skipped; CTest's SKIP_RETURN_CODE).
// CHECK records a failure and carries on; REQUIRE stops the test.

#include <cstdio>
#include <cstdlib>
#include <string>

#include <cuda_runtime.h>

#include "wsl_cuda_gl.h"

namespace harness {

inline int& failures() {
    static int n = 0;
    return n;
}

inline void section(const char* name) { std::printf("--- %s\n", name); }

[[noreturn]] inline void skip(const std::string& why) {
    std::printf("SKIPPED: %s\n", why.c_str());
    std::exit(77);
}

inline int finish() {
    if (failures()) {
        std::printf("FAILED: %d check(s)\n", failures());
        return 1;
    }
    std::printf("PASSED\n");
    return 0;
}

} // namespace harness

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            ++harness::failures();                                                        \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                                 \
    } while (0)

#define REQUIRE(cond)                                                                       \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::fprintf(stderr, "%s:%d: REQUIRE failed: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                                   \
        }                                                                                   \
    } while (0)

// Checks that `expr` throws wslcg::Error whose message contains `substring`.
#define CHECK_THROWS(expr, substring)                                                                    \
    do {                                                                                                 \
        bool threw_ = false;                                                                             \
        try {                                                                                            \
            (void)(expr);                                                                                \
        } catch (const wslcg::Error& e_) {                                                               \
            threw_ = true;                                                                               \
            if (std::string(e_.what()).find(substring) == std::string::npos) {                           \
                ++harness::failures();                                                                   \
                std::fprintf(stderr, "%s:%d: %s threw \"%s\", expected it to mention \"%s\"\n", __FILE__, \
                             __LINE__, #expr, e_.what(), substring);                                     \
            }                                                                                            \
        }                                                                                                \
        if (!threw_) {                                                                                   \
            ++harness::failures();                                                                       \
            std::fprintf(stderr, "%s:%d: %s did not throw wslcg::Error\n", __FILE__, __LINE__, #expr);    \
        }                                                                                                \
    } while (0)

#define CUDA_REQUIRE(expr)                                                                          \
    do {                                                                                            \
        cudaError_t err_ = (expr);                                                                  \
        if (err_ != cudaSuccess) {                                                                  \
            std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #expr, cudaGetErrorString(err_)); \
            std::exit(1);                                                                           \
        }                                                                                           \
    } while (0)
