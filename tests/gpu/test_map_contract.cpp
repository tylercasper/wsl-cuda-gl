// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

// The map/unmap state machine and move semantics, mirroring CUDA-GL interop's rules.

#include <utility>

#include "gl_env.h"
#include "harness.h"

int main() {
    HeadlessGL gl;
    require_gpu_environment(gl);
    uint8_t* d_mem = nullptr;
    CUDA_REQUIRE(cudaMalloc(&d_mem, 64));

    harness::section("map / unmap");
    wslcg::Buffer buf(4096);
    CHECK(!buf.is_mapped());
    CHECK_THROWS(buf.unmap(), "not mapped");
    CHECK_THROWS(buf.copy_from_device(d_mem, 4), "mapped");
    CHECK_THROWS(buf.copy_to_device(d_mem, 4), "mapped");
    void* p1 = buf.map();
    CHECK(p1 != nullptr);
    CHECK(buf.is_mapped());
    CHECK_THROWS(buf.map(), "already mapped");
    buf.unmap();
    CHECK(!buf.is_mapped());
    void* p2 = buf.map();
    CHECK(p1 == p2); // the same memory every time
    buf.unmap();

    harness::section("streams");
    cudaStream_t stream;
    CUDA_REQUIRE(cudaStreamCreate(&stream));
    buf.map();
    buf.copy_from_device(d_mem, 64, 0, stream);
    buf.unmap(stream);
    buf.map();
    buf.unmap(cudaStreamPerThread);
    CUDA_REQUIRE(cudaStreamDestroy(stream));

    harness::section("size");
    CHECK_THROWS(wslcg::Buffer(0), "greater than 0");
    wslcg::Buffer tiny(1);
    CHECK(tiny.size() == 1);

    harness::section("move");
    wslcg::Buffer a(8192);
    const unsigned name = a.gl_buffer();
    wslcg::Buffer b(std::move(a));
    CHECK(b.gl_buffer() == name && b.size() == 8192);
    CHECK(a.gl_buffer() == 0 && a.size() == 0 && !a.is_mapped());
    CHECK_THROWS(a.map(), "moved-from");
    CHECK_THROWS(a.unmap(), "moved-from");
    b.map();
    b = wslcg::Buffer(16384); // releases the mapped 8192-byte buffer
    CHECK(b.size() == 16384 && !b.is_mapped());
    CHECK(!glIsBuffer(name));
    a = std::move(b);
    CHECK(a.size() == 16384 && b.size() == 0);

    CUDA_REQUIRE(cudaFree(d_mem));
    return harness::finish();
}
