// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

// copy_from_device / copy_to_device at every alignment their kernels distinguish
// (16-byte, 4-byte, byte), checked from the GL side and from CUDA.

#include <algorithm>
#include <vector>

#include "gl_env.h"
#include "harness.h"

int main() {
    HeadlessGL gl;
    require_gpu_environment(gl);

    const size_t size = (4u << 20) + 64;
    wslcg::Buffer buf(size);
    std::vector<uint8_t> pattern(size);
    for (size_t i = 0; i < size; ++i) pattern[i] = (uint8_t)(i * 7 + 3);
    uint8_t* d_src = nullptr;
    uint8_t* d_dst = nullptr;
    CUDA_REQUIRE(cudaMalloc(&d_src, size));
    CUDA_REQUIRE(cudaMalloc(&d_dst, size));
    CUDA_REQUIRE(cudaMemcpy(d_src, pattern.data(), size, cudaMemcpyHostToDevice));

    // {offset into the buffer, offset into the source, bytes}
    const size_t cases[][3] = {{0, 0, 1},       {3, 5, 7},         {0, 0, 4},         {4, 8, 64},
                               {16, 32, 1 << 20}, {1, 0, 1000003}, {0, 1, 4 << 20}, {0, 0, size}};
    for (auto& c : cases) {
        size_t off = c[0], src_off = c[1], bytes = c[2];
        std::printf("--- %zu bytes at buffer offset %zu from source offset %zu\n", bytes, off, src_off);

        buf.map();
        buf.copy_from_device(d_src + src_off, bytes, off, nullptr);
        buf.unmap(nullptr);
        std::vector<uint8_t> via_gl(bytes);
        glGetNamedBufferSubData(buf.gl_buffer(), (GLintptr)off, (GLsizeiptr)bytes, via_gl.data());
        CHECK(std::equal(via_gl.begin(), via_gl.end(), pattern.begin() + src_off));

        CUDA_REQUIRE(cudaMemset(d_dst, 0, size));
        buf.map();
        buf.copy_to_device(d_dst + src_off, bytes, off, nullptr);
        buf.unmap(nullptr);
        std::vector<uint8_t> via_cuda(bytes);
        CUDA_REQUIRE(cudaMemcpy(via_cuda.data(), d_dst + src_off, bytes, cudaMemcpyDeviceToHost));
        CHECK(std::equal(via_cuda.begin(), via_cuda.end(), pattern.begin() + src_off));
    }

    harness::section("bounds");
    buf.map();
    CHECK_THROWS(buf.copy_from_device(d_src, 2, size - 1), "exceeds");
    CHECK_THROWS(buf.copy_to_device(d_dst, 1, size), "exceeds");
    CHECK_THROWS(buf.copy_from_device(d_src, 1, (size_t)-1), "exceeds");
    buf.copy_from_device(d_src, 0, size); // empty copy at the end is fine
    buf.unmap(nullptr);

    CUDA_REQUIRE(cudaFree(d_src));
    CUDA_REQUIRE(cudaFree(d_dst));
    return harness::finish();
}
