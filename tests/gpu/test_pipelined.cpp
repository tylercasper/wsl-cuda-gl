// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

// Two buffers in flight: CUDA renders frame N (behind a simulated workload) while GL
// draws frame N-1. Every displayed frame must be exactly the frame it should be, with
// nothing from the frame being rendered at the same time.

#include "gl_env.h"
#include "harness.h"
#include "pattern.cuh"

int main() {
    HeadlessGL gl;
    require_gpu_environment(gl);

    const int w = 640, h = 360, frames = 40;
    wslcg::Buffer bufs[2] = {wslcg::Buffer((size_t)w * h * 4), wslcg::Buffer((size_t)w * h * 4)};
    Fbo fbo(w, h);
    fbo.bind();
    GLuint prog = make_buffer_blit_program(w, h);
    cudaStream_t stream;
    CUDA_REQUIRE(cudaStreamCreate(&stream));

    size_t bad_frames = 0;
    for (int frame = 0; frame <= frames; ++frame) {
        wslcg::Buffer& next = bufs[frame % 2];
        wslcg::Buffer& prev = bufs[(frame + 1) % 2];
        if (frame < frames) {
            auto* px = static_cast<uint32_t*>(next.map());
            launch_busy_wait(2000, stream); // CUDA still busy while GL draws below
            launch_pattern_linear(px, w, w, h, frame, stream);
            CUDA_REQUIRE(cudaGetLastError());
        }
        if (frame > 0) {
            glUseProgram(prog);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, prev.gl_buffer());
            glDrawArrays(GL_TRIANGLES, 0, 3);
            size_t bad = pattern_mismatches(fbo.read_top_down(), w, h, frame - 1);
            if (bad) std::fprintf(stderr, "frame %d: %zu pixels wrong\n", frame - 1, bad);
            bad_frames += bad != 0;
        }
        if (frame < frames) next.unmap(stream);
    }
    std::printf("%d frames displayed, %zu wrong\n", frames, bad_frames);
    CHECK(bad_frames == 0);
    CUDA_REQUIRE(cudaStreamDestroy(stream));
    glDeleteProgram(prog);
    return harness::finish();
}
