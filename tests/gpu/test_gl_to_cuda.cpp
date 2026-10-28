// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

// GL writes (a compute shader), CUDA reads: both through a user kernel on the mapped
// pointer and through copy_to_device. There is deliberately no glFinish between the
// dispatch and map(): map() itself must wait for GL.

#include <vector>

#include "gl_env.h"
#include "harness.h"
#include "pattern.cuh"

static size_t hash_mismatches(const std::vector<uint32_t>& v, size_t first) {
    size_t bad = 0;
    for (size_t i = 0; i < v.size(); ++i)
        if (v[i] != hash_u32((uint32_t)(first + i)) && bad++ < 3)
            std::fprintf(stderr, "  word %zu: got %08x want %08x\n", first + i, v[i], hash_u32((uint32_t)(first + i)));
    return bad;
}

int main() {
    HeadlessGL gl;
    require_gpu_environment(gl);

    const size_t n = 16u << 20; // 64 MiB: long enough on the GPU that a missing wait would show
    wslcg::Buffer buf(n * 4);
    GLuint prog = compile_compute_program(R"(#version 450 core
        layout(local_size_x = 256) in;
        layout(std430, binding = 0) writeonly buffer Words { uint v[]; };
        layout(location = 0) uniform uint u_n;
        void main() {
            uint stride = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
            for (uint i = gl_GlobalInvocationID.x; i < u_n; i += stride) v[i] = (i * 2654435761u) ^ (i >> 7);
        })");

    uint32_t* d_out = nullptr;
    CUDA_REQUIRE(cudaMalloc(&d_out, n * 4));
    std::vector<uint32_t> host(n);

    for (int round = 0; round < 3; ++round) {
        std::printf("--- round %d\n", round);
        CUDA_REQUIRE(cudaMemset(d_out, 0, n * 4));
        glUseProgram(prog);
        glProgramUniform1ui(prog, 0, (GLuint)n);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf.gl_buffer());
        glDispatchCompute(4096, 1, 1); // grid-stride: GL only guarantees 65535 groups per dimension
        CHECK(glGetError() == GL_NO_ERROR);

        // A user kernel reading the mapped pointer.
        auto* px = static_cast<uint32_t*>(buf.map());
        launch_copy_u32(d_out, px, n, nullptr);
        CUDA_REQUIRE(cudaGetLastError());
        buf.unmap(nullptr);
        CUDA_REQUIRE(cudaMemcpy(host.data(), d_out, n * 4, cudaMemcpyDeviceToHost));
        size_t bad = hash_mismatches(host, 0);
        std::printf("user kernel read: %zu of %zu words differ\n", bad, n);
        CHECK(bad == 0);

        // The library's copy, from an offset.
        const size_t first = 12345, count = n - 2 * first;
        CUDA_REQUIRE(cudaMemset(d_out, 0, n * 4));
        buf.map();
        buf.copy_to_device(d_out, count * 4, first * 4, nullptr);
        buf.unmap(nullptr);
        std::vector<uint32_t> part(count);
        CUDA_REQUIRE(cudaMemcpy(part.data(), d_out, count * 4, cudaMemcpyDeviceToHost));
        bad = hash_mismatches(part, first);
        std::printf("copy_to_device: %zu of %zu words differ\n", bad, count);
        CHECK(bad == 0);
    }
    CUDA_REQUIRE(cudaFree(d_out));
    glDeleteProgram(prog);
    return harness::finish();
}
