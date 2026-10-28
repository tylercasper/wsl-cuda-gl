// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

// map() must wait for GL and unmap() must wait for CUDA. Checked two ways:
//
//  * the contract, deterministically: when map() returns, GL work issued before it
//    has finished; when unmap(stream) returns, the stream is idle;
//  * the consequence: no torn or stale data. Quick GPU work hides a missing wait (work
//    from the two APIs tends to run in submission order anyway), so the producer here
//    runs for ~150-200 ms, long enough for the GPU scheduler to start the consumer in
//    the middle of it. (On the RTX 4090 / driver 615 this catches a missing unmap()
//    wait; CUDA never preempts the long GL pass there, so a missing map() wait is
//    caught by the contract check only.)
//
// Verified to fail with either wait removed from the library.

#include <chrono>
#include <vector>

#include "gl_env.h"
#include "harness.h"
#include "pattern.cuh"

static double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

// unmap(): GL must not see the buffer until CUDA's slow write has finished.
static void unmap_waits_for_cuda() {
    harness::section("unmap() waits for CUDA");
    const int w = 512, h = 256;
    wslcg::Buffer buf((size_t)w * h * 4);
    Fbo fbo(w, h);
    fbo.bind();
    GLuint prog = make_buffer_blit_program(w, h);
    glUseProgram(prog);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf.gl_buffer());

    for (int frame = 1; frame <= 3; ++frame) {
        auto* px = static_cast<uint32_t*>(buf.map());
        launch_pattern_linear(px, w, w, h, 0, nullptr); // the stale frame
        launch_busy_wait(200000, nullptr);
        launch_pattern_linear(px, w, w, h, frame, nullptr); // the frame GL must show
        CUDA_REQUIRE(cudaGetLastError());
        auto t = std::chrono::steady_clock::now();
        buf.unmap(nullptr);
        std::printf("unmap took %.0f ms\n", seconds_since(t) * 1000);
        CHECK(cudaStreamQuery(nullptr) == cudaSuccess); // contract: CUDA is done
        glDrawArrays(GL_TRIANGLES, 0, 3);
        size_t bad = pattern_mismatches(fbo.read_top_down(), w, h, frame);
        std::printf("frame %d: %zu pixels not from the finished CUDA frame\n", frame, bad);
        CHECK(bad == 0);
    }
    glDeleteProgram(prog);
}

// map(): CUDA must not read the buffer until GL's slow write has finished.
static void map_waits_for_gl() {
    harness::section("map() waits for GL");
    const size_t n = 1u << 20;
    wslcg::Buffer buf(n * 4);
    // Each invocation spins `u_spin` iterations before writing its word. The branch
    // depends on the loop result, so the loop can't be optimized away; it is never taken.
    GLuint prog = compile_compute_program(R"(#version 450 core
        layout(local_size_x = 256) in;
        layout(std430, binding = 0) writeonly buffer Words { uint v[]; };
        layout(location = 0) uniform uint u_n;
        layout(location = 1) uniform uint u_spin;
        layout(location = 2) uniform uint u_seed;
        void main() {
            uint stride = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
            for (uint i = gl_GlobalInvocationID.x; i < u_n; i += stride) {
                uint acc = i;
                for (uint k = 0u; k < u_spin; ++k) acc = acc * 1664525u + 1013904223u;
                v[i] = (acc == 0xFFFFFFFFu && u_spin == 7u) ? 0u : (((i * 2654435761u) ^ (i >> 7)) + u_seed);
            }
        })");
    glUseProgram(prog);
    glProgramUniform1ui(prog, 0, (GLuint)n);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf.gl_buffer());

    // Calibrate the spin count to ~150 ms on this GPU.
    unsigned spin = 1000;
    for (;;) {
        glProgramUniform1ui(prog, 1, spin);
        glFinish();
        auto t = std::chrono::steady_clock::now();
        glDispatchCompute(64, 1, 1);
        glFinish();
        double s = seconds_since(t);
        if (s > 0.15 || spin > (1u << 30)) break;
        spin = (unsigned)(spin * (s > 0.01 ? 0.15 / s : 10.0)) + 1;
    }
    std::printf("GL compute pass calibrated to %u spins\n", spin);
    CHECK(glGetError() == GL_NO_ERROR);

    uint32_t* d_out = nullptr;
    CUDA_REQUIRE(cudaMalloc(&d_out, n * 4));
    std::vector<uint32_t> host(n);
    for (uint32_t seed = 1; seed <= 3; ++seed) {
        glProgramUniform1ui(prog, 2, seed);
        glDispatchCompute(64, 1, 1); // ~150 ms, not waited for here
        GLsync issued_before_map = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        auto t = std::chrono::steady_clock::now();
        auto* px = static_cast<uint32_t*>(buf.map());
        std::printf("map took %.0f ms\n", seconds_since(t) * 1000);
        GLint status = 0;
        glGetSynciv(issued_before_map, GL_SYNC_STATUS, 1, nullptr, &status);
        CHECK(status == GL_SIGNALED); // contract: GL is done
        glDeleteSync(issued_before_map);
        launch_copy_u32(d_out, px, n, nullptr);
        CUDA_REQUIRE(cudaGetLastError());
        buf.unmap(nullptr);
        CUDA_REQUIRE(cudaMemcpy(host.data(), d_out, n * 4, cudaMemcpyDeviceToHost));
        size_t bad = 0;
        for (size_t i = 0; i < n; ++i) bad += host[i] != hash_u32((uint32_t)i) + seed;
        std::printf("pass %u: %zu of %zu words not from the finished GL pass\n", seed, bad, n);
        CHECK(bad == 0);
    }
    CUDA_REQUIRE(cudaFree(d_out));
    glDeleteProgram(prog);
}

int main() {
    HeadlessGL gl;
    require_gpu_environment(gl);
    unmap_waits_for_cuda();
    map_waits_for_gl();
    return harness::finish();
}
