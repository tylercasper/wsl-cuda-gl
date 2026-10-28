// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Headless GL for the tests (no window or display server needed) and the GL helpers
// they share.

#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <GL/glext.h>

#include <cstdint>
#include <string>
#include <vector>

// A GL 4.5 core context on EGL's surfaceless platform, current on the creating thread.
class HeadlessGL {
public:
    HeadlessGL() = default;
    ~HeadlessGL();
    HeadlessGL(const HeadlessGL&) = delete;
    HeadlessGL& operator=(const HeadlessGL&) = delete;

    // False, with the reason in *why, if this machine can't provide one.
    bool create(std::string* why);
    std::string renderer() const;

private:
    void* display_ = nullptr;
    void* context_ = nullptr;
};

// The GPU tests need WSL2 with /dev/dxg, a CUDA device, and Mesa's d3d12 driver on that
// GPU. If the machine lacks any of them this skips the test (exit 77), unless
// WSLCG_REQUIRE_GPU_TESTS=1 is set, which turns a skip into a failure so a GPU machine
// can't silently skip. On success `gl` is current and CUDA's primary context exists.
void require_gpu_environment(HeadlessGL& gl);

GLuint compile_program(const char* fragment_source);
GLuint compile_compute_program(const char* source);

// Draws a w x h image stored as packed RGBA8 uints (row 0 = top) in the SSBO at
// binding 0, 1:1 into the viewport.
GLuint make_buffer_blit_program(int w, int h);

// An RGBA8 framebuffer object to render into and read back.
class Fbo {
public:
    Fbo(int w, int h);
    ~Fbo();
    void bind() const;
    // The framebuffer's pixels with row 0 the TOP row.
    std::vector<uint32_t> read_top_down() const;

private:
    int w_, h_;
    GLuint fbo_ = 0, tex_ = 0;
};

// Pixels of a top-down w x h image that differ from pattern_rgba(..., frame);
// prints the first few.
size_t pattern_mismatches(const std::vector<uint32_t>& top_down, int w, int h, int frame);
