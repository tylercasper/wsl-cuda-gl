// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Test image and helper kernels for the examples, tests and probe.
//
// The image is integer-only on purpose: the CPU reproduces it bit-exactly, so the
// tests can prove that the pixels GL displays are exactly the ones CUDA wrote.

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#ifdef __CUDACC__
#define WSLCG_HD __host__ __device__
#else
#define WSLCG_HD
#endif

WSLCG_HD inline int tri_wave(int t, int period) {
    if (period <= 0) return 0; // tiny images
    int m = t % (2 * period);
    return m < period ? m : 2 * period - m;
}

// Pixel (x, y) of frame `frame`, with y = 0 the TOP row. Packed RGBA8, R in the low
// byte (GL_RGBA / GL_UNSIGNED_BYTE order).
WSLCG_HD inline uint32_t pattern_rgba(int x, int y, int w, int h, int frame) {
    int cx1 = w / 4 + tri_wave(frame * 4, w / 2), cy1 = h / 3 + tri_wave(frame * 3, h / 3);
    int cx2 = 3 * w / 4 - tri_wave(frame * 5, w / 2), cy2 = 2 * h / 3 - tri_wave(frame * 2, h / 3);
    int dx1 = x - cx1, dy1 = y - cy1, dx2 = x - cx2, dy2 = y - cy2;
    uint32_t r = ((dx1 * dx1 + dy1 * dy1) >> 8) & 0xff;
    uint32_t g = ((dx2 * dx2 + dy2 * dy2) >> 8) & 0xff;
    uint32_t b = ((x + frame * 2) ^ (y - frame)) & 0xff;
    return r | (g << 8) | (b << 16) | (0xffu << 24);
}

// Word i of the sequence GL writes in the GL -> CUDA tests (same formula in GLSL).
WSLCG_HD inline uint32_t hash_u32(uint32_t i) { return (i * 2654435761u) ^ (i >> 7); }

// Writes the test image; `pitch_px` is the row stride in pixels, row 0 is the top.
void launch_pattern_linear(uint32_t* dst, int pitch_px, int w, int h, int frame, cudaStream_t s);
// Occupies `s` for `us` microseconds of GPU time, standing in for a real workload.
void launch_busy_wait(int us, cudaStream_t s);
// A user kernel reading from (or writing to) a mapped buffer: dst[i] = src[i].
void launch_copy_u32(uint32_t* dst, const uint32_t* src, size_t n, cudaStream_t s);
