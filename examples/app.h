// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Scaffolding shared by the examples: window/context creation, display shaders,
// verification and per-phase timing. Nothing here moves frame data between CUDA and
// GL; that is what each example (and the library) shows.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <GL/glext.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (expr);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #expr,          \
                    cudaGetErrorString(err_));                                            \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

#define CHECK(cond, ...)                                                                  \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            fprintf(stderr, "%s:%d: check failed: %s: ", __FILE__, __LINE__, #cond);      \
            fprintf(stderr, __VA_ARGS__);                                                 \
            fprintf(stderr, "\n");                                                        \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

struct Options {
    int width = 1280, height = 720;
    int frames = 0;      // 0 = run until the window is closed
    bool hidden = false; // don't map the window (for automated runs)
    bool verify = false; // read back the last frame and compare to the CPU reference
    bool vsync = false;
    std::string dump;    // write the last displayed frame to this .ppm
    std::string display = "ssbo"; // zero-copy examples: "ssbo" or "texture"
    int cuda_work_us = 0;         // simulated CUDA workload per frame, before rendering
};
Options parse_options(int argc, char** argv);

struct App {
    GLFWwindow* window = nullptr;
    Options opt;
};

// Points Mesa at its D3D12 driver on CUDA's GPU (wslcg::configure_mesa), creates a
// GL 4.5 core context and hands glfwGetProcAddress to the library.
App app_init(const Options& opt, const char* title);
bool app_running(App& app, int frames_done);
void app_shutdown(App& app);

double now_ms();


GLuint compile_program(const char* fs_source);

// Shader that draws a w x h RGBA8 sampler2D 1:1 into the viewport, flipping so
// that texel row 0 (the top image row in this project) ends up at the top.
GLuint make_texture_blit_program();

// Shader that draws a w x h image stored as packed RGBA8 uints (row 0 = top) in the
// shader storage buffer at binding 0, 1:1 into the viewport. Reads the buffer in
// place: no copy.
GLuint make_buffer_blit_program(int w, int h);

// Read back the back buffer and compare against pattern_rgba() for `frame`.
// Verification only: this readback is not part of any step's data path.
bool verify_backbuffer(int w, int h, int frame, const std::string& dump_path);

// Accumulates per-phase times, ignoring the first `warmup` frames (CUDA module
// loading, first-use allocations, shader compiles).
class PhaseTimer {
public:
    explicit PhaseTimer(int warmup = 10) : warmup_(warmup) {}
    void add(const char* name, double ms);
    void end_frame() { ++frames_; }
    void report() const;

private:
    int warmup_, frames_ = 0;
    std::vector<std::pair<std::string, double>> totals_;
};
